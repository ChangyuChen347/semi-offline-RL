import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from . import register_model
from config.decorator import replace
from transformers.models.bart.modeling_bart import *
from transformers.file_utils import ModelOutput
from models.loss import *
import random
import numpy as np
from numpy import random as np_rand
from torch.distributions import Categorical
import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

class CustomBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.is_scoring_mode = False
    def scoring_mode(self):
        self.is_scoring_mode = True
    def generation_mode(self):
        self.is_scoring_mode = False
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        import time
        ste = time.time()
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            # decoder_input_ids = shift_tokens_right(
            #     input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            # )
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, 0
            )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        import time
        st = time.time()
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        if self.is_scoring_mode:
            if len(decoder_input_ids.shape) != 3:
                cand_num = decoder_input_ids.size(0) // encoder_outputs[0].size(0)
            else:
                cand_num = decoder_input_ids.size(1)
            encoder_hidden_states = encoder_outputs[0]
            encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, cand_num, dim=0)
            if attention_mask is not None:
                attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            else:
                cand_mask = decoder_input_ids != 1
                cand_mask[:, 0] = 1
                decoder_attention_mask = cand_mask.view(-1, cand_mask.size(-1))
        else:
            encoder_hidden_states = encoder_outputs[0]

        st = time.time()
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs
        res = Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        return res

@replace(BartForConditionalGeneration)
class BartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()
        self.config = config
        try:
            self.mask_input = config.mask_input
            self.mask_rate = config.mask_rate
            self.tokenizer_name = config.tokenizer_name
            self.model_parallel = False
            self.device_map = None
            self.vocab_size = config.vocab_size
            self.mask_id = 50264
            self.sample_topk = config.sample_topk
        except AttributeError as e:
            print(e)
            pass

    def get_masked_token(self, tk_id, cont):
        return self.mask_id
    def forward (self,**kwargs):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            pos = kwargs["input_ids"] == self.mask_id
            e_id = 50118
            kwargs["input_ids"] = pos * torch.ones_like(kwargs["input_ids"]).long().cuda() * e_id + ~pos * kwargs["input_ids"]
        if not self.training and "past_key_values" not in kwargs and 'not_seq_decode' not in kwargs:
            bs = kwargs["input_ids"].shape[0]
            res = self.generate(
                input_ids = kwargs["input_ids"],
                attention_mask = kwargs["attention_mask"],
            )
            pad = res.new_full(list(res.shape[:-1])+[max(self.config.max_length - res.shape[-1],0)],0)
            res = torch.cat([res,pad],dim=-1)
            res = res.view([bs,-1] + list(res.shape[1:]))
            return {"loss":pad.new_full([1],0.0,dtype=torch.float32),"ids":res}
        elif 'normal_forward' in kwargs:
            kwargs.pop('normal_forward')
            if 'doc' in kwargs:
                kwargs.pop('doc') #todo
            if 'query' in kwargs:
                kwargs.pop('query') #todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'labels' in kwargs:
                kwargs.pop('labels')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            import time
            st = time.time()
            res = super(BartForConditionalGeneration, self).forward(**kwargs)
            return res
        else:
            if 'doc' in kwargs:
                kwargs.pop('doc') #todo
            if 'query' in kwargs:
                kwargs.pop('query') #todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            import time
            st = time.time()
            res = self.rl_forward(**kwargs)
            return res

    def rl_forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        mask_labels=None,
        masked_pos_shift=None,
        masked_pos_non_shift=None,
        non_masked_pos_shift=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        import time
        st = time.time()
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bs = None
        if labels is not None:
            bs = labels.shape[0]
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                bs = labels.shape[0]
                if mask_labels is None:
                    mask_labels = labels.detach().cpu().clone()  # bs, seq
                mask_labels_np = mask_labels.numpy()
                def get_tks(y):
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
                        self.tokenizer.convert_ids_to_tokens(output_ids[i]) for i in
                        range(len(output_ids))]
                    return tks
                tmp_tks = get_tks(mask_labels_np)
                non_zero_labels = ~(
                        labels.data.eq(1) | labels.data.eq(2) | labels.data.eq(-100))  # 0 pad 2 eos -100 pad
                non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
                non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()
                non_masked_pos_shift = torch.zeros_like(mask_labels)
                if masked_pos_shift is None:
                    masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                    masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
                    labels_numpy = mask_labels_np
                    should_mask_pos = np.zeros(labels_numpy.shape)
                    labels_numpy = labels_numpy.tolist()
                    for i in range(bs):
                        if input_ids.shape[0] != bs:
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
                            if tmp_tks[i][k][0] != 'Ġ' and tmp_tks[i][k] != '.' and tmp_tks[i][k] != ',':  # if pre is mask this is not mask it will connect
                                should_mask_pos[i][k] = 1
                            else:
                                should_mask_pos[i][k] = 0
                            if self.config.cand_pos_remove_sp_tk and spmask:
                                if self.config.nmask_next_comma:
                                    if k > 0 and (
                                            tmp_tks[i][k - 1] == ',' or tmp_tks[i][k - 1] == '.' or tmp_tks[i][
                                        k - 1] == 'Ġ,' or tmp_tks[i][k - 1] == 'Ġ.'):
                                        k += 1
                                        continue
                            if tmp_tks[i][k][0] != 'Ġ' and tmp_tks[i][k] != '.' and tmp_tks[i][k] != ',':  # if pre is mask this is not mask it will connect
                                should_mask_pos[i][k] = 1
                            else:
                                should_mask_pos[i][k] = 0
                            cand_pos.append(k)
                            k += 1

                        if not self.config.fix_mask:
                            sample_num = 0
                            for _ in range(len(cand_pos)):
                                if random.random() < self.mask_rate:
                                    sample_num += 1
                        else:
                            sample_num = int(len(cand_pos) * self.mask_rate)

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
                        cont = 0
                        for idx, j in enumerate(sample_pos):
                            if idx > 0 and sample_pos[idx - 1] + 1 == j:
                                cont += 1
                            else:
                                cont = 0
                            if self.mask_input:
                                mask_labels[i][j] = self.get_masked_token(mask_labels[i][j], cont)
                            masked_pos_shift[i][idx] = j + 1
                            masked_pos_non_shift[i][idx] = j
                        for idx, j in enumerate(non_sample_pos):
                            if random.random() <= 1:
                                non_masked_pos_shift[i][idx] = j

                decoder_input_ids = shift_tokens_right(
                    mask_labels, self.config.pad_token_id, 0
                )

        decoder_input_ids = decoder_input_ids.cuda()

        st = time.time()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits2 = None


        def construct_return(lm_logits, labels, bs, masked_pos_shift, non_masked_pos_shift, masked_pos_non_shift, ce=False):
            if masked_pos_non_shift is not None:
                topk = self.sample_topk
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
                mp = torch.zeros((bs, seq_len+1)).cuda().scatter_(1, masked_pos_shift.cuda(), torch.ones((bs, seq_len+1)).cuda())
                mp = mp[:, 1:]
                mp_long = mp.long()
                ori_mp = mp.clone()
                ones = torch.ones_like(labels, dtype=torch.long).cuda()

                other_2_pad_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(2)) + ones * (labels.data.eq(-100) | labels.data.eq(2)) # -100 -> 0#
                ori_other_2_pad_labels = other_2_pad_labels.clone()
                pads = ones.clone()
                _, max_ids = torch.max(log_probs, dim=-1)
                eos_mask = labels == 2
                eos_log_probs = log_probs.gather(2, torch.ones_like(labels).long().cuda().unsqueeze(2)*2).squeeze()
                eos_log_probs = eos_log_probs * eos_mask
                y_b = other_2_pad_labels * (1-mp_long) + max_ids * mp_long
                y_zero_b = pads * (1-mp_long) + max_ids * mp_long
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
                    probs = torch.softmax(mask*logits, -1)
                    ori_masked_ids = masked_ids
                    prob = torch.gather(probs, dim=1, index=masked_ids)
                    masked_ids = masked_ids.reshape(-1, s2, sample_num + 1).transpose(1,2)
                    prob = prob.reshape(-1, s2, sample_num + 1).transpose(1,2)
                    log_probs = torch.log(prob)
                    masked_ids = masked_ids.reshape(bs * (sample_num + 1), s2)
                    log_probs = log_probs.reshape(bs * (sample_num + 1), s2)

                    other_2_pad_labels = other_2_pad_labels.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    other_2_pad_labels = other_2_pad_labels.reshape(bs * (sample_num + 1), -1)
                    mp_long = mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp_long = mp_long.reshape(bs * (sample_num + 1), -1)
                    mp = mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp = mp.reshape(bs * (sample_num + 1), -1)

                    pads = pads.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    pads = pads.reshape(bs * (sample_num + 1), -1)
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
                    other_2_pad_labels = other_2_pad_labels.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    other_2_pad_labels = other_2_pad_labels.reshape(bs * (sample_num + 1), -1)
                    mp_long = mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp_long = mp_long.reshape(bs * (sample_num + 1), -1)
                    mp = mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp = mp.reshape(bs * (sample_num + 1), -1)
                    pads = pads.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    pads = pads.reshape(bs * (sample_num + 1), -1)
                y_s = other_2_pad_labels * (1 - mp_long) + masked_ids * mp_long
                y_zero_s = pads * (1 - mp_long) + masked_ids * mp_long
                y_zero_labels = pads * (1 - mp_long) + other_2_pad_labels * mp_long

                if sample_num != 0:
                    log_probs = log_probs * mp.float()

            masked_lm_loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))

            if not return_dict:
                output = (lm_logits,) + outputs[1:]
                return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

            if labels is None:
                return Seq2SeqLMOutput(
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
            else:
                return Seq2SeqLMOutput(
                    loss=masked_lm_loss,
                    logits=lm_logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions,
                ), y_b, y_s, max_ids, masked_ids, input_ids, labels, non_zero_sum_tensor, \
                       log_probs, y_zero_b, y_zero_s, y_zero_labels, None, log_probs_all, lm_logits, \
                       mask_labels, masked_pos_shift, \
                       masked_pos_non_shift, decoder_input_ids, None, None, None, None, None, eos_log_probs
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        if lm_logits is not None and input_ids is not None and lm_logits.shape[0] != input_ids.shape[0]:
            # print(lm_logits.shape)
            lm_logits = lm_logits.reshape(-1, 2, lm_logits.shape[-2], lm_logits.shape[-1])
            ce_lm_logits = lm_logits[:, 0, :]
            lm_logits = lm_logits[:, 1, :]

            ce_labels = labels.reshape(-1, 2, labels.shape[-1])
            labels = ce_labels[:, 1, :]
            ce_labels = ce_labels[:, 0, :]

            # labels = labels.reshape(-1, 2, labels.shape[-1])
            masked_pos_shift = masked_pos_shift.reshape(-1, 2, masked_pos_shift.shape[-1])
            ce_masked_pos_shift = masked_pos_shift[:, 0, :]
            masked_pos_shift = masked_pos_shift[:, 1, :]
            masked_pos_non_shift = masked_pos_non_shift.reshape(-1, 2, masked_pos_non_shift.shape[-1])
            ce_masked_pos_non_shift = masked_pos_non_shift[:, 0, :]
            masked_pos_non_shift = masked_pos_non_shift[:, 1, :]
            non_masked_pos_shift = non_masked_pos_shift.reshape(-1, 2, non_masked_pos_shift.shape[-1])
            ce_non_masked_pos_shift = non_masked_pos_shift[:, 0, :]
            non_masked_pos_shift = non_masked_pos_shift[:, 1, :]
            res= [construct_return(lm_logits=ce_lm_logits, labels=ce_labels, bs=bs//2,
                                    non_masked_pos_shift=ce_non_masked_pos_shift, masked_pos_shift=ce_masked_pos_shift, masked_pos_non_shift=ce_masked_pos_non_shift, ce=True),
                    construct_return(lm_logits=lm_logits, labels=labels, bs=bs // 2,
                                     non_masked_pos_shift=non_masked_pos_shift, masked_pos_shift=masked_pos_shift,
                                     masked_pos_non_shift=masked_pos_non_shift),

                    ]
        else:
            res= construct_return(lm_logits=lm_logits, labels=labels, bs=bs, non_masked_pos_shift=non_masked_pos_shift,
                                    masked_pos_shift=masked_pos_shift, masked_pos_non_shift=masked_pos_non_shift)
        return res




