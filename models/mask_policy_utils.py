import random
import numpy as np
from numpy import random as np_rand
import torch
import torch.nn as nn
from transformers.modeling_outputs import (

    Seq2SeqLMOutput,

)

class MaskPolicy():

    def __init__(self, mask_id, config, bpe_prefix):
        self.mask_id = mask_id
        self.config = config
        self.bpe_prefix = bpe_prefix


    def get_masked_input(self, labels, encoder_input_bs, tokenizer):
        bs = labels.shape[0]
        mask_labels = labels.detach().cpu().clone()  # bs, seq
        mask_labels_np = mask_labels.numpy()

        tmp_tks = self.get_tks(mask_labels_np, tokenizer)

        non_zero_labels = ~(
                labels.data.eq(tokenizer.pad_token_id) | labels.data.eq(tokenizer.eos_token_id) | labels.data.eq(-100))  # 0 pad 2 eos -100 pad
        # the count of non-special token
        non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
        non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()

        masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
        masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
        labels_numpy = mask_labels_np
        should_mask_pos = np.zeros(labels_numpy.shape)
        labels_numpy = labels_numpy.tolist()
        for i in range(bs):
            if encoder_input_bs != bs: #
                if i % 2 == 0:
                    spmask = False
                else:
                    spmask = True
            else:
                spmask = True
            label_np = labels_numpy[i]
            cand_pos = []
            k = 0
            while k < non_zero_sum[i]:
                if label_np[k] == 0:
                    k += 1
                    continue
                if tmp_tks[i][k][0] != self.bpe_prefix and tmp_tks[i][k] != '.' and tmp_tks[i][
                    k] != ',':  # if the previous token is [mask], we should mask the whole span
                    should_mask_pos[i][k] = 1
                else:
                    should_mask_pos[i][k] = 0
                if self.config.cand_pos_remove_sp_tk and spmask:
                    if self.config.nmask_next_comma:
                        if k > 0 and (
                                tmp_tks[i][k - 1] == ',' or tmp_tks[i][k - 1] == '.' or tmp_tks[i][
                            k - 1] == self.bpe_prefix+',' or tmp_tks[i][k - 1] == self.bpe_prefix+'.'):
                            k += 1
                            continue
                cand_pos.append(k)
                k += 1

            if not self.config.fix_mask:
                sample_num = 0
                for _ in range(len(cand_pos)):
                    if random.random() < self.config.mask_rate:
                        sample_num += 1
            else:
                sample_num = int(len(cand_pos) * self.config.mask_rate)

            sample_pos = np_rand.choice(a=np.array(cand_pos), size=sample_num, replace=False).tolist()

            if self.config.span_mask and spmask:
                extra_sample_pos = []
                for pos in sample_pos:
                    tmp_pos = pos
                    while tmp_pos + 1 < len(should_mask_pos[i]) and should_mask_pos[i][tmp_pos + 1] == 1:
                        extra_sample_pos.append(tmp_pos + 1)
                        tmp_pos += 1
                sample_pos = list(set(sample_pos) | set(extra_sample_pos))

            sample_pos = sorted(sample_pos)
            non_sample_pos = set(cand_pos) - set(sample_pos)
            non_sample_pos = sorted(list(non_sample_pos))

            for idx, j in enumerate(sample_pos):
                if self.config.mask_input:
                    mask_labels[i][j] = self.mask_id
                masked_pos_shift[i][idx] = j + 1
                masked_pos_non_shift[i][idx] = j


        return mask_labels, masked_pos_shift, masked_pos_non_shift

    def get_tks(self, y, tokenizer):
        pre_output_ids = y.tolist()
        output_ids = []
        for i in range(len(pre_output_ids)):
            output_id = []
            for j in range(0, len(pre_output_ids[i])):
                if pre_output_ids[i][j] == -100:
                    break
                output_id.append(pre_output_ids[i][j])
            output_ids.append(output_id)
        tks = [
            tokenizer.convert_ids_to_tokens(output_ids[i]) for i in
            range(len(output_ids))]
        return tks

    def construct_return(self, lm_logits, labels, bs, masked_pos_shift, masked_pos_non_shift, outputs,
                         input_ids,  mask_labels, decoder_input_ids, tokenizer, ce=False, return_dict=True):
        '''

        '''
        if masked_pos_non_shift is not None:
            topk = self.config.sample_topk
            if topk == -1:
                probs = torch.softmax(lm_logits, dim=-1)
            else:
                indices_to_remove = lm_logits < torch.topk(lm_logits, topk)[0][..., -1, None]
                indices_to_keep = lm_logits >= torch.topk(lm_logits, topk)[0][..., -1, None]
                to_sample_lm_logits = lm_logits * indices_to_keep.cuda() + indices_to_remove.cuda() * torch.ones_like(
                    lm_logits).cuda() * -1e8
                to_sample_probs = torch.softmax(to_sample_lm_logits, dim=-1)
                probs = torch.softmax(lm_logits, dim=-1)
            log_probs = torch.log(probs)
            log_probs_all = log_probs.detach().clone()

            seq_len = labels.shape[1]
            mp = torch.zeros((bs, seq_len + 1)).cuda().scatter_(1, masked_pos_shift.cuda(),
                                                                torch.ones((bs, seq_len + 1)).cuda())
            mp = mp[:, 1:]
            mp_long = mp.long()
            ori_mp = mp.clone()

            pads = torch.ones_like(labels, dtype=torch.long).cuda() * tokenizer.pad_token_id
            other_2_pads_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(tokenizer.eos_token_id)) + pads * (
                        labels.data.eq(-100) | labels.data.eq(tokenizer.eos_token_id))  # -100 -> 0#
            _, max_ids = torch.max(log_probs, dim=-1)
            y_b = other_2_pads_labels * (1 - mp_long) + max_ids * mp_long

            sample_num = 0
            if not ce:
                sample_num = self.config.sample_num
            if sample_num != 0:
                _, s2, s3 = probs.shape
                probs = probs.reshape(-1, s3)
                logits = lm_logits.reshape(-1, s3)
                masked_ids = torch.multinomial(probs, sample_num + 1, replacement=True)
                # bs * seq_len, sample_num
                mask = torch.zeros_like(logits).cuda().long().scatter_(1, masked_ids,
                                                                       torch.ones_like(masked_ids).long().cuda())
                probs = torch.softmax(mask * logits, -1)
                ori_masked_ids = masked_ids
                prob = torch.gather(probs, dim=1, index=masked_ids)
                masked_ids = masked_ids.reshape(-1, s2, sample_num + 1).transpose(1, 2)
                prob = prob.reshape(-1, s2, sample_num + 1).transpose(1, 2)
                log_probs = torch.log(prob)
                masked_ids = masked_ids.reshape(bs * (sample_num + 1), s2)
                log_probs = log_probs.reshape(bs * (sample_num + 1), s2)

                other_2_pads_labels = other_2_pads_labels.unsqueeze(1).expand(-1, sample_num + 1, -1)
                other_2_pads_labels = other_2_pads_labels.reshape(bs * (sample_num + 1), -1)
                mp_long = mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                mp_long = mp_long.reshape(bs * (sample_num + 1), -1)
                mp = mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                mp = mp.reshape(bs * (sample_num + 1), -1)
            else:
                _, s2, s3 = probs.shape
                probs = probs.reshape(-1, s3)
                if self.config.use_logit:
                    logits = lm_logits.reshape(-1, s3)
                masked_ids = torch.multinomial(probs, sample_num + 1, replacement=True)
                mask = torch.zeros_like(logits).cuda().long().scatter_(1, masked_ids,
                                                                       torch.ones_like(masked_ids).long().cuda())
                probs = torch.softmax(mask * logits, -1)
                prob = torch.gather(probs, dim=1, index=masked_ids)
                masked_ids = masked_ids.reshape(-1, s2, sample_num + 1).transpose(1, 2)
                prob = prob.reshape(-1, s2, sample_num + 1).transpose(1, 2)
                log_probs = torch.log(prob)
                masked_ids = masked_ids.reshape(bs * (sample_num + 1), s2)
                log_probs = log_probs.reshape(bs * (sample_num + 1), s2)
                other_2_pads_labels = other_2_pads_labels.unsqueeze(1).expand(-1, sample_num + 1, -1)
                other_2_pads_labels = other_2_pads_labels.reshape(bs * (sample_num + 1), -1)
                mp_long = mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                mp_long = mp_long.reshape(bs * (sample_num + 1), -1)
                mp = mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                mp = mp.reshape(bs * (sample_num + 1), -1)

            y_s = other_2_pads_labels * (1 - mp_long) + masked_ids * mp_long
            if sample_num != 0:
                log_probs = log_probs * mp.float()

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            raise NotImplementedError("not return_dict not implemented yet.")
            # output = (lm_logits,) + outputs[1:]
            # return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        output = Seq2SeqLMOutput(
                loss=masked_lm_loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
                encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                encoder_hidden_states=outputs.encoder_hidden_states,
                encoder_attentions=outputs.encoder_attentions,
            )

        if labels is None:
            return output
        else:
            return output, y_b, y_s, max_ids, masked_ids, input_ids, labels,log_probs, log_probs_all, \
                mask_labels.to(input_ids.device), masked_pos_shift.to(input_ids.device), \
                masked_pos_non_shift.to(input_ids.device), decoder_input_ids

    def split_return(self, lm_logits, input_ids, labels,masked_pos_shift,
                     masked_pos_non_shift, bs, return_dict, outputs, mask_labels, decoder_input_ids, tokenizer):
        '''
        split the output for mle and rl
        '''
        if lm_logits is not None and input_ids is not None and lm_logits.shape[0] != input_ids.shape[0]:
            lm_logits = lm_logits.reshape(-1, 2, lm_logits.shape[-2], lm_logits.shape[-1])
            ce_lm_logits = lm_logits[:, 0, :]
            lm_logits = lm_logits[:, 1, :]

            ce_labels = labels.reshape(-1, 2, labels.shape[-1])
            labels = ce_labels[:, 1, :]
            ce_labels = ce_labels[:, 0, :]

            masked_pos_shift = masked_pos_shift.reshape(-1, 2, masked_pos_shift.shape[-1])
            ce_masked_pos_shift = masked_pos_shift[:, 0, :]
            masked_pos_shift = masked_pos_shift[:, 1, :]

            masked_pos_non_shift = masked_pos_non_shift.reshape(-1, 2, masked_pos_non_shift.shape[-1])
            ce_masked_pos_non_shift = masked_pos_non_shift[:, 0, :]
            masked_pos_non_shift = masked_pos_non_shift[:, 1, :]

            'not split "outputs" yet.'
            res= [self.construct_return(lm_logits=ce_lm_logits, labels=ce_labels, bs=bs//2,
                                     masked_pos_shift=ce_masked_pos_shift,
                                   masked_pos_non_shift=ce_masked_pos_non_shift,
                                   ce=True, return_dict=return_dict, outputs=outputs, input_ids=input_ids,
                                                 mask_labels=mask_labels, decoder_input_ids=decoder_input_ids, tokenizer=tokenizer
                                                 ),
                    self.construct_return(lm_logits=lm_logits, labels=labels, bs=bs // 2,
                                      masked_pos_shift=masked_pos_shift,
                                     masked_pos_non_shift=masked_pos_non_shift, return_dict=return_dict, outputs=outputs, input_ids=input_ids,
                                                 mask_labels=mask_labels, decoder_input_ids=decoder_input_ids, tokenizer=tokenizer),

                    ]
        else:
            res= self.construct_return(lm_logits=lm_logits, labels=labels, bs=bs,
                                    masked_pos_shift=masked_pos_shift, masked_pos_non_shift=masked_pos_non_shift, return_dict=return_dict,
                                                outputs=outputs, input_ids=input_ids,
                                                 mask_labels=mask_labels, decoder_input_ids=decoder_input_ids, tokenizer=tokenizer)

        return res