
import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from config.decorator import replace
from transformers.models.pegasus.modeling_pegasus import *
import numpy as np
from numpy import random as np_rand
from torch.distributions import Categorical
import torch.nn.functional as F
from transformers.file_utils import ModelOutput
import random
from transformers.models.pegasus.configuration_pegasus import PegasusConfig
try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive."""

        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, input):
            return input


@replace(PegasusForConditionalGeneration)
class PegasusForConditionalGeneration(PegasusForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomPegasusModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.init_weights()

        self.config = config
        self.mask_input = config.mask_input
        self.mask_rate = config.mask_rate
        self.out_mask_rate = config.out_mask_rate
        self.io_not_same_mask = config.io_not_same_mask
        self.tokenizer_name = config.tokenizer_name
        self.model_parallel = False
        self.device_map = None
        self.do_parallel_test_model = config.do_parallel_test_model
        self.inv_out_mask = config.inv_out_mask
        self.tail_unmask_num = config.tail_unmask_num
        self.truth_log_probs = config.truth_log_probs
        self.is_scoring_mode = False
        self.vocab_size = config.vocab_size
        self.use_cont_mask_id = config.use_cont_mask_id
        self.mask_id = 3
        self.span_mask = config.span_mask
        self.keep_prob = config.keep_prob
        self.random_prob = config.random_prob
        self.not_mask_stop = config.not_mask_stop
        self.sample_topk = config.sample_topk

    # def scoring_mode(self):
    #     self.is_scoring_mode = True
    #
    # def generation_mode(self):
    #     self.is_scoring_mode = False

    def forward (self,**kwargs):
        if not self.training and "past_key_values" not in kwargs and 'not_seq_decode' not in kwargs:
            bs = kwargs["input_ids"].shape[0]
            res = self.generate(
                input_ids = kwargs["input_ids"],
                attention_mask = kwargs["attention_mask"]
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
            return self.normal_forward(**kwargs)
        else:
            if 'doc' in kwargs:
                kwargs.pop('doc') #todo
            if 'query' in kwargs:
                kwargs.pop('query') #todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            #return super().forward(**kwargs)
            return self.rl_forward(**kwargs)

    def normal_forward(
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
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

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


    def get_masked_token(self, tk_id, cont):
        p = random.random()
        # self.keep_prob = 0.1
        # self.random_prob = 0.1
        if p < self.keep_prob:
            return tk_id
        elif p < self.keep_prob + self.random_prob:
            return random.randint(0, self.vocab_size - 1)
        else:
            if not self.use_cont_mask_id:
                return self.mask_id
            else:
                return self.mask_id + cont



    def rl_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
            mask_labels=None,
            masked_pos_shift=None,
            masked_pos_non_shift=None,
            non_masked_pos_shift=None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mask_decoder_inputs = True
        masked_pos_non_shift = None
        bs = None
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            bs = labels.shape[0]
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
            mask_id = 3
            if mask_decoder_inputs:
                bs = labels.shape[0]
                mask_labels = labels.clone()  # bs, seq

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

                tmp_tks = get_tks(mask_labels.clone().detach().cpu().numpy())
                masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                non_masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
                non_zero_labels = ~(
                            labels.data.eq(0) | labels.data.eq(1) | labels.data.eq(-100))  # 0 pad 1 <\s> -100 pad
                # 1 eos 0 pad -100 ce pad
                non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
                non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()
                labels_numpy = mask_labels.detach().cpu().numpy()
                should_mask_pos = np.zeros(labels_numpy.shape)
                labels_numpy = labels_numpy.tolist()
                v2 = self.config.v2_0401
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
                    while k < non_zero_sum[i] - self.tail_unmask_num:
                        if not v2:
                            if tmp_tks[i][k][0] != '▁':  # if pre is mask this is not mask it will connect
                                should_mask_pos[i][k] = 1
                            else:
                                should_mask_pos[i][k] = 0

                            if self.config.cand_pos_remove_sp_tk:
                                if k >= 1 and (
                                        tmp_tks[i][k-1] == ',' or tmp_tks[i][k-1] == '.'):
                                    k += 1
                                    continue
                                if self.not_mask_stop:
                                    if tmp_tks[i][k] == '▁' or tmp_tks[i][k] == ',' or tmp_tks[i][k] == '.':
                                        k += 1
                                        continue
                        else:
                            if tmp_tks[i][k][0] != '▁' and tmp_tks[i][k] != '.' and tmp_tks[i][k] != ',':  # if pre is mask this is not mask it will connect
                                should_mask_pos[i][k] = 1
                            else:
                                should_mask_pos[i][k] = 0
                            if self.config.cand_pos_remove_sp_tk and spmask:
                                # . x  .后面的词不mask
                                if self.config.nmask_comma:
                                    if (tmp_tks[i][k] == ',' or tmp_tks[i][k] == '.'):
                                        k += 1
                                        continue
                                if self.config.nmask_next_comma:
                                    if k > 0 and (tmp_tks[i][k - 1] == ',' or tmp_tks[i][k - 1] == '.'):
                                        k += 1
                                        continue
                        cand_pos.append(k)
                        k += 1
                    sample_num = int(len(cand_pos) * self.mask_rate)

                    sample_pos = np_rand.choice(a=np.array(cand_pos), size=sample_num, replace=False).tolist()

                    extra_sample_pos = []
                    if v2:
                        if self.span_mask and spmask:
                            for pos in sample_pos:
                                tmp_pos = pos
                                while tmp_pos + 1 < len(should_mask_pos[i]) and should_mask_pos[i][tmp_pos + 1] == 1:
                                    extra_sample_pos.append(tmp_pos + 1)
                                    tmp_pos += 1
                    else:
                        if self.span_mask:
                            for pos in sample_pos:
                                tmp_pos = pos
                                while tmp_pos + 1 < len(should_mask_pos[i]) and should_mask_pos[i][tmp_pos + 1] == 1:
                                    extra_sample_pos.append(tmp_pos + 1)
                                    tmp_pos += 1
                    sample_pos = list(set(sample_pos) | set(extra_sample_pos))
                    sample_pos = sorted(sample_pos)
                    non_sample_pos = set(cand_pos) - set(sample_pos)
                    non_sample_pos = sorted(list(non_sample_pos))
                    if self.io_not_same_mask:
                        cont = 0
                        for idx, j in enumerate(sample_pos):
                            if idx > 1 and sample_pos[idx-1] + 1 == j:
                                cont += 1
                            else:
                                cont = 0
                            if self.mask_input:
                                mask_labels[i][j] = self.get_masked_token(mask_labels[i][j], cont)
                        out_sample_num = int(len(cand_pos) * self.out_mask_rate)
                        if self.inv_out_mask:
                            sample_pos_set = set(sample_pos)
                            sample_pos2 = [t for t in cand_pos if t not in sample_pos_set]
                        else:
                            sample_pos2 = np_rand.choice(a=np.array(cand_pos), size=out_sample_num,
                                                         replace=False).tolist()
                        sample_pos2 = sorted(sample_pos2)
                        for idx, j in enumerate(sample_pos2):
                            masked_pos_shift[i][idx] = j + 1
                            masked_pos_non_shift[i][idx] = j
                    else:
                        cont = 0
                        for idx, j in enumerate(sample_pos):
                            if idx > 1 and sample_pos[idx - 1] + 1 == j:
                                cont += 1
                            else:
                                cont = 0
                            if self.mask_input:
                                mask_labels[i][j] = self.get_masked_token(mask_labels[i][j], cont)
                            masked_pos_shift[i][idx] = j + 1
                            masked_pos_non_shift[i][idx] = j
                        for idx, j in enumerate(non_sample_pos):
                            if random.random() < 0.5:
                                non_masked_pos_shift[i][idx] = j
                #decoder_input_ids = self._shift_right(mask_labels)  # 0, 1, 2  pred 1, 2,
                decoder_input_ids = shift_tokens_right(
                        mask_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                    )
            else:
                decoder_input_ids = shift_tokens_right(
                            labels, self.config.pad_token_id, self.config.decoder_start_token_id
                        )

        # Decode
     #   print(decoder_input_ids)
        decoder_outputs = self.model(
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




        def construct_return(lm_logits, labels, bs, masked_pos_shift, non_masked_pos_shift, masked_pos_non_shift,
                             ce=False):

            if mask_decoder_inputs and masked_pos_non_shift is not None:
                topk = self.config.sample_topk
                if topk == -1:
                    to_sample_lm_logits = lm_logits
                else:
                    indices_to_remove = lm_logits < torch.topk(lm_logits, topk)[0][..., -1, None]
                    indices_to_keep = lm_logits >= torch.topk(lm_logits, topk)[0][..., -1, None]
                    # print(indices_to_remove.shape)
                    # print(probs.shape)
                    to_sample_lm_logits = lm_logits * indices_to_keep.cuda() + indices_to_remove.cuda() * torch.ones_like(
                        lm_logits).cuda() * -1e8

                to_sample_probs = torch.softmax(to_sample_lm_logits, dim=-1)
                # to_sample_probs[:, :, 1] = 0
                probs = torch.softmax(lm_logits, dim=-1)
                log_probs = torch.log(probs+ 1e-8)

                log_probs_all = log_probs.detach().clone()
                seq_len = labels.shape[1]
                mp = torch.zeros((bs, seq_len+1)).cuda().scatter_(1, masked_pos_shift, torch.ones((bs, seq_len+1)).cuda())
                mp = mp[:, 1:]
                mp_long = mp.long()
                pad_2_zero_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(1))  # -100 -> 0#

                zeros = torch.zeros_like(labels, dtype=torch.long).cuda()
                pads = zeros.clone()
                log_probs_m = log_probs.detach().clone()
                # log_probs_m[:, :, 1] = -1e8
                _, max_ids = torch.max(log_probs_m, dim=-1)
                # y_b = labels.clone()
                # y_b = torch.cat([torch.zeros(bs, 1, dtype=torch.long).cuda(), y_b], dim=-1)
                # y_b.scatter_(1, masked_pos_shift, max_ids)

                y_b = pad_2_zero_labels * (1-mp_long) + max_ids * mp_long
                y_zero_b = pads * (1-mp_long) + max_ids * mp_long

                # non_mp = torch.zeros((bs, seq_len + 1)).cuda().scatter_(1, non_masked_pos_shift,
                #                                                         torch.ones((bs, seq_len + 1)).cuda())
                # non_mp = non_mp[:, 1:]
                # non_mp_long = non_mp.long()

                y_b_g = pad_2_zero_labels * (1 - mp_long) + max_ids * mp_long

                # y_b_g = y_b_g * (1 - non_mp_long) + max_ids * non_mp_long

                y_b_g_m = y_b_g * (1 - mp_long) + torch.ones_like(max_ids).long().cuda() * self.mask_id * mp_long

                if self.config.sample_num != 0:
                    # to_sample_probs = to_sample_probs.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1, -1)

                    if self.config.sample_method == 'multi':
                        _, s2, s3 = probs.shape
                        probs = probs.reshape(-1, s3)
                        if self.config.use_logit:
                            logits = lm_logits.reshape(-1, s3)
                        masked_ids = torch.multinomial(probs, self.config.sample_num + 1, replacement=True)
                        # bs * seq_len, sample_num
                        if self.config.use_logit:
                            # logit = torch.gather(logit, dim=1, index=masked_ids) #bs, seq_len, sample_num
                            # mask = torch.zeros_like(logit).cuda().long().scatter_(1, masked_ids,torch.ones_like(masked_ids).long().cuda())

                            mask = torch.zeros_like(logits).cuda().long().scatter_(1, masked_ids,
                                                                                   torch.ones_like(
                                                                                       masked_ids).long().cuda())
                            # print(probs[0])

                            probs = torch.softmax(mask * logits, -1)

                            # print(probs[0])
                        prob = torch.gather(probs, dim=1, index=masked_ids)
                        if self.config.prob_w:
                            prob_w = torch.gather(probs.detach().clone(), dim=1, index=masked_ids)

                        masked_ids = masked_ids.reshape(-1, s2, self.config.sample_num + 1).transpose(1, 2)
                        if self.config.prob_w:
                            prob_w = prob_w.reshape(-1, s2, self.config.sample_num + 1).transpose(1, 2)
                        prob = prob.reshape(-1, s2, self.config.sample_num + 1).transpose(1, 2)
                        log_probs = torch.log(prob)
                        if self.config.prob_w:
                            log_probs = log_probs * prob_w
                        masked_ids = masked_ids.reshape(bs * (self.config.sample_num + 1), s2)
                        log_probs = log_probs.reshape(bs * (self.config.sample_num + 1), s2)

                        pad_2_zero_labels = pad_2_zero_labels.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        pad_2_zero_labels = pad_2_zero_labels.reshape(bs * (self.config.sample_num + 1), -1)
                        mp_long = mp_long.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        mp_long = mp_long.reshape(bs * (self.config.sample_num + 1), -1)
                        mp = mp.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        mp = mp.reshape(bs * (self.config.sample_num + 1), -1)

                        # non_mp_long = non_mp_long.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        # non_mp_long = non_mp_long.reshape(bs * (self.config.sample_num + 1), -1)
                        # non_mp = non_mp.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        # non_mp = non_mp.reshape(bs * (self.config.sample_num + 1), -1)
                        pads = pads.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        pads = pads.reshape(bs * (self.config.sample_num + 1), -1)

                    # log_probs = log_probs / ((mp_long.sum(-1) + 1e-8) ** 0.6).unsqueeze(1).unsqueeze(1)
                    # log_probs = log_probs * 0.01


                else:
                    _, s2, s3 = probs.shape
                    probs = probs.reshape(-1, s3)
                    if self.config.use_logit:
                        logits = lm_logits.reshape(-1, s3)
                    masked_ids = torch.multinomial(probs, self.config.sample_num + 1, replacement=True)
                    # bs * seq_len, sample_num
                    if self.config.use_logit:
                        # logit = torch.gather(logit, dim=1, index=masked_ids) #bs, seq_len, sample_num
                        # mask = torch.zeros_like(logit).cuda().long().scatter_(1, masked_ids,torch.ones_like(masked_ids).long().cuda())

                        mask = torch.zeros_like(logits).cuda().long().scatter_(1, masked_ids,
                                                                               torch.ones_like(
                                                                                   masked_ids).long().cuda())
                        # print(probs[0])

                        probs = torch.softmax(mask * logits, -1)

                        # print(probs[0])
                    prob = torch.gather(probs, dim=1, index=masked_ids)
                    if self.config.prob_w:
                        prob_w = torch.gather(probs.detach().clone(), dim=1, index=masked_ids)

                    masked_ids = masked_ids.reshape(-1, s2, self.config.sample_num + 1).transpose(1, 2)
                    if self.config.prob_w:
                        prob_w = prob_w.reshape(-1, s2, self.config.sample_num + 1).transpose(1, 2)
                    prob = prob.reshape(-1, s2, self.config.sample_num + 1).transpose(1, 2)
                    log_probs = torch.log(prob)
                    if self.config.prob_w:
                        log_probs = log_probs * prob_w
                    masked_ids = masked_ids.reshape(bs * (self.config.sample_num + 1), s2)
                    log_probs = log_probs.reshape(bs * (self.config.sample_num + 1), s2)

                    pad_2_zero_labels = pad_2_zero_labels.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                    pad_2_zero_labels = pad_2_zero_labels.reshape(bs * (self.config.sample_num + 1), -1)
                    mp_long = mp_long.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                    mp_long = mp_long.reshape(bs * (self.config.sample_num + 1), -1)
                    mp = mp.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                    mp = mp.reshape(bs * (self.config.sample_num + 1), -1)

                    # non_mp_long = non_mp_long.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                    # non_mp_long = non_mp_long.reshape(bs * (self.config.sample_num + 1), -1)
                    # non_mp = non_mp.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                    # non_mp = non_mp.reshape(bs * (self.config.sample_num + 1), -1)
                    pads = pads.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                    pads = pads.reshape(bs * (self.config.sample_num + 1), -1)
                # y_zero_b = torch.zeros_like(labels, dtype=torch.long).cuda()
                # y_zero_b = torch.cat([torch.zeros(bs, 1, dtype=torch.long).cuda(), y_zero_b], dim=-1)
                # y_zero_b.scatter_(1, masked_pos_shift, max_ids)



                y_s = pad_2_zero_labels * (1 - mp_long) + masked_ids * mp_long
                y_zero_s = pads * (1 - mp_long) + masked_ids * mp_long
                y_zero_labels = pads * (1 - mp_long) + pad_2_zero_labels * mp_long



                y_s_g = pad_2_zero_labels * (1 - mp_long) + masked_ids * mp_long

                # y_s_g = y_s_g * (1 - non_mp_long) + masked_ids * non_mp_long

                y_s_g_m = y_s_g * (1 - mp_long) + torch.ones_like(masked_ids).long().cuda() * self.mask_id * mp_long

                # y_s = labels.clone()
                # y_s = torch.cat([torch.zeros(bs, 1, dtype=torch.long).cuda(), y_s], dim=-1)
                # y_s.scatter_(1, masked_pos_shift, masked_ids)
                # y_zero_s = torch.zeros_like(labels, dtype=torch.long).cuda()
                # y_zero_s = torch.cat([torch.zeros(bs, 1, dtype=torch.long).cuda(), y_zero_s], dim=-1)
                # y_zero_s.scatter_(1, masked_pos_shift, masked_ids)
                # y_zero_labels = torch.zeros_like(labels, dtype=torch.long).cuda()
                # y_zero_labels = torch.cat([torch.zeros(bs, 1, dtype=torch.long).cuda(), y_zero_labels], dim=-1)
                # y_zero_labels.scatter_(1, masked_pos_shift, labels_masked)

                if self.truth_log_probs:
                    truth_log_probs = log_probs.clone().gather(2, pad_2_zero_labels.unsqueeze(2)).squeeze()
                else:
                    truth_log_probs = None

                # if self.config.cand_num != 0 and self.config.sample_num == 0:
                #     masked_probs = log_probs.gather(2, y_s.unsqueeze(2)).squeeze()
                #     # print(labels)
                #     mp_all = ~(labels.data.eq(-100) | labels.data.eq(1))
                #     mp_all = mp_all.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                #     mp_all = mp_all.reshape(bs * (self.config.sample_num + 1), -1)
                #     log_probs = masked_probs * mp_all.float()
                if self.config.sample_num != 0 and self.config.sample_method == 'multi':
                    log_probs = log_probs * mp.float()
                elif self.config.sample_num != 0 and self.config.sample_method == 'loop':
                    if self.config.use_all_probs:
                        masked_probs = log_probs.gather(2, y_s.unsqueeze(2)).squeeze()
                        # print(labels)
                        mp_all = ~(labels.data.eq(-100) | labels.data.eq(1))
                        mp_all = mp_all.unsqueeze(1).expand(-1, self.config.sample_num + 1, -1)
                        mp_all = mp_all.reshape(bs * (self.config.sample_num + 1), -1)
                        log_probs = masked_probs * mp_all.float()
                    else:
                        # masked_probs = log_probs.gather(2, masked_ids.unsqueeze(2)).squeeze()
                        # masked_probs = log_probs.gather(2, masked_ids.unsqueeze(2)).squeeze()
                        #
                        # log_probs = masked_probs * mp.float()

                        log_probs = log_probs * mp.float()

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1))
                # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
                return ((loss,) + output) if loss is not None else output
            if labels is None:
                return Seq2SeqLMOutput(
                    loss=loss,
                    logits=lm_logits,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.decoder_hidden_states,
                    decoder_attentions=decoder_outputs.decoder_attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
                    encoder_hidden_states=decoder_outputs.encoder_hidden_states,
                    encoder_attentions=decoder_outputs.encoder_attentions,
                )

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.decoder_hidden_states,
                decoder_attentions=decoder_outputs.decoder_attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=decoder_outputs.encoder_last_hidden_state,
                encoder_hidden_states=decoder_outputs.encoder_hidden_states,
                encoder_attentions=decoder_outputs.encoder_attentions,
            ), y_b, y_s, max_ids, masked_ids, input_ids, labels, non_zero_sum_tensor, log_probs, y_zero_b, y_zero_s, y_zero_labels, truth_log_probs, log_probs_all, lm_logits,  mask_labels, masked_pos_shift, masked_pos_non_shift, decoder_input_ids, y_b_g, y_b_g_m, y_s_g, y_s_g_m

        '''
        '''
        sequence_output = decoder_outputs[0]

        lm_logits = self.lm_head(sequence_output) + self.final_logits_bias  # bs, seq_len,
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

            res = [construct_return(lm_logits=ce_lm_logits, labels=ce_labels, bs=bs // 2,
                                    non_masked_pos_shift=ce_non_masked_pos_shift, masked_pos_shift=ce_masked_pos_shift,
                                    masked_pos_non_shift=ce_masked_pos_non_shift, ce=True),
                   construct_return(lm_logits=lm_logits, labels=labels, bs=bs // 2,
                                    non_masked_pos_shift=non_masked_pos_shift, masked_pos_shift=masked_pos_shift,
                                    masked_pos_non_shift=masked_pos_non_shift),

                   ]


        else:
            res = construct_return(lm_logits=lm_logits, labels=labels, bs=bs, non_masked_pos_shift=non_masked_pos_shift,
                                   masked_pos_shift=masked_pos_shift, masked_pos_non_shift=masked_pos_non_shift)
        return res
        # masked_lm_loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        #
        # if not return_dict:
        #     output = (lm_logits,) + outputs[1:]
        #     return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


class CustomPegasusModel(PegasusModel):
    def __init__(self, config: PegasusConfig):
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
        r"""
        Returns:

        Example::

            >>> from transformers import PegasusTokenizer, PegasusModel

            >>> tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
            >>> model = PegasusModel.from_pretrained("google/pegasus-large")

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

            >>> last_hidden_states = outputs.last_hidden_state
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            # cand_num = decoder_input_ids.size(1)
            hidden_states = encoder_outputs[0]
            hidden_states = torch.repeat_interleave(hidden_states, cand_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            else:
                cand_mask = decoder_input_ids != 0
                cand_mask[:, 0] = 1
                decoder_attention_mask = cand_mask.view(-1, cand_mask.size(-1))
            # print(attention_mask.shape)
            # print(decoder_input_ids.shape)
            # print(decoder_attention_mask.shape)
        else:
            hidden_states = encoder_outputs[0]
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        #print(decoder_input_ids.shape)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=hidden_states,
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

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )