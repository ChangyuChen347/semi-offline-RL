import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from config.decorator import replace
from transformers.models.t5.modeling_t5 import *
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
import random
import numpy as np
from numpy import random as np_rand
from torch.distributions import Categorical

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


class RLSeq2SeqLMOutput(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    y_b: Optional[torch.LongTensor] = None
    y_s: Optional[torch.LongTensor] = None
    max_ids: Optional[torch.LongTensor] = None
    masked_ids: Optional[torch.LongTensor] = None
    input_ids: Optional[torch.LongTensor] = None
    labels: Optional[torch.LongTensor] = None
    non_zero_sum_tensor: Optional[torch.LongTensor] = None
    log_probs: Optional[torch.FloatTensor] = None
    y_zero_b: Optional[torch.LongTensor] = None
    y_zero_s: Optional[torch.LongTensor] = None
    y_zero_labels: Optional[torch.LongTensor] = None
    truth_log_probs: Optional[torch.FloatTensor] = None
    log_probs_all: Optional[torch.FloatTensor] = None
    lm_logits: Optional[torch.FloatTensor] = None

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



@replace(T5ForConditionalGeneration)
class T5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        try:
            self.mask_input = config.mask_input
            self.mask_rate = config.mask_rate
            self.out_mask_rate = config.out_mask_rate
            self.io_not_same_mask = config.io_not_same_mask
            self.tokenizer_name = config.tokenizer_name

            self.device_map = None

            self.truth_log_probs = config.truth_log_probs
            self.is_scoring_mode = False
            self.do_rl = config.do_rl
        except AttributeError as e:
            print(e)
            self.is_debug=True
            pass
    def scoring_mode(self):
        self.is_scoring_mode = True

    def generation_mode(self):
        self.is_scoring_mode = False

    def normal_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        #hidden_states = encoder_outputs[0]




        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        if self.is_scoring_mode:
            cand_num = decoder_input_ids.size(1)
            hidden_states = encoder_outputs[0]
            hidden_states = torch.repeat_interleave(hidden_states, cand_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
            decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
        else:
            hidden_states = encoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

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
            query=None,
            doc=None,
            q2=None,
            rl_agent=None,
            mask_labels=None,
            masked_pos_shift=None,
            masked_pos_non_shift=None,

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        Returns:
        Examples::
            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        mask_decoder_inputs = True

        bs = None
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            bs = labels.shape[0]
            if mask_decoder_inputs:
                bs = labels.shape[0]
                non_zero_labels = ~(
                        labels.data.eq(0) | labels.data.eq(1) | labels.data.eq(-100))  # 0 pad 1 <\s> -100 pad
                non_zero_sum_tensor = non_zero_labels.sum(-1)  # b
                non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()

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
                    masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                    masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
                    labels_numpy = mask_labels_np
                    should_mask_pos = np.zeros(labels_numpy.shape)
                    labels_numpy = labels_numpy.tolist()
                    for i in range(bs):
                        label_np = labels_numpy[i]
                        cand_pos = []
                        k = 0
                        while k < non_zero_sum[i]:
                            if tmp_tks[i][k][0] != '▁':  # if the previous token is [mask], we should mask the whole span
                                should_mask_pos[i][k] = 1
                            else:
                                should_mask_pos[i][k] = 0

                            if self.config.cand_pos_remove_sp_tk:
                                if label_np[k] == 250100 or label_np[k] == 250101:
                                    k += 1
                                    continue
                                if k+1 < len(label_np) and (label_np[k+1] == 250100 or label_np[k+1] == 250101):
                                    k += 1
                                    continue
                                if k>=1 and (label_np[k-1] == 260 or label_np[k-1] == 261 or label_np[k-1] == 250100 or label_np[k-1] == 250101):
                                    k += 1
                                    continue

                            cand_pos.append(k)
                            k += 1

                        def get_masked_token(tk_id):
                            return self.mask_id

                        sample_num = int(len(cand_pos) * self.mask_rate)
                        sample_pos = np_rand.choice(a=np.array(cand_pos), size=sample_num, replace=False).tolist()
                        if self.config.span_mask:
                            extra_sample_pos = []
                            for pos in sample_pos:
                                tmp_pos = pos
                                while tmp_pos + 1 < len(should_mask_pos[i]) and should_mask_pos[i][tmp_pos + 1] == 1:
                                    extra_sample_pos.append(tmp_pos+1)
                                    tmp_pos += 1

                            sample_pos = list(set(sample_pos) | set(extra_sample_pos))
                        sample_pos = sorted(sample_pos)

                        non_sample_pos = set(cand_pos) - set(sample_pos)
                        non_sample_pos = sorted(list(non_sample_pos))

                        if self.tokenizer_name == 't5-small':
                            pad_st_id = 32000
                            pad_ed_id = 32099
                            self.vocab_size = 32100
                            self.mask_id = pad_ed_id
                        else:
                            pad_st_id = 250000
                            pad_ed_id = 250099
                            self.vocab_size = 250100
                            self.mask_id = pad_ed_id
                        if self.io_not_same_mask:
                            for idx, j in enumerate(sample_pos):
                                if self.mask_input:
                                    mask_labels[i][j] = get_masked_token(mask_labels[i][j])
                            out_sample_num = int(len(cand_pos) * self.out_mask_rate)

                            sample_pos2 = np_rand.choice(a=np.array(cand_pos), size=out_sample_num,
                                                             replace=False).tolist()
                            sample_pos2 = sorted(sample_pos2)
                            for idx, j in enumerate(sample_pos2):
                                masked_pos_shift[i][idx] = j + 1
                                masked_pos_non_shift[i][idx] = j
                        else:
                            tot = 0
                            for idx, j in enumerate(sample_pos):
                                if self.mask_input:
                                    mask_labels[i][j] = get_masked_token(mask_labels[i][j])
                                masked_pos_shift[i][idx] = j + 1
                                masked_pos_non_shift[i][idx] = j


                decoder_input_ids = self._shift_right(mask_labels)  # 0, 1, 2  pred 1, 2,
            else:
                decoder_input_ids = self._shift_right(labels)

        decoder_input_ids = decoder_input_ids.cuda()

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        if self.is_scoring_mode:
            # cand_num = decoder_input_ids.size(1)
            if len(decoder_input_ids.shape) != 3:
                cand_num = decoder_input_ids.size(0) // encoder_outputs[0].size(0)
            else:
                cand_num = decoder_input_ids.size(1)
            hidden_states = encoder_outputs[0]
            hidden_states = torch.repeat_interleave(hidden_states, cand_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
            # decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            else:
                cand_mask = decoder_input_ids != 0
                cand_mask[:, 0] = 1
                decoder_attention_mask = cand_mask.view(-1, cand_mask.size(-1))
        else:
            hidden_states = encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        lm_logits = self.lm_head(sequence_output)  # bs, seq_len, dim -> bs, seq_len, vocab


        predict_baseline = None

        def construct_return(lm_logits, labels, bs, masked_pos_shift,  masked_pos_non_shift,
                             ce=False):

            if mask_decoder_inputs and masked_pos_non_shift is not None:
                probs = torch.softmax(lm_logits, dim=-1)
                topk = self.config.sample_topk
                if topk == -1:
                    to_sample_lm_logits = lm_logits.clone().detach()
                else:
                    indices_to_remove = lm_logits < torch.topk(lm_logits, topk)[0][..., -1, None]
                    indices_to_keep = lm_logits >= torch.topk(lm_logits, topk)[0][..., -1, None]
                    to_sample_lm_logits = lm_logits * indices_to_keep.cuda() + indices_to_remove.cuda() * torch.ones_like(
                        lm_logits).cuda() * -1e8
                    to_sample_lm_logits = to_sample_lm_logits.clone().detach()
                to_sample_probs = torch.softmax(to_sample_lm_logits, dim=-1)

                '''
                topk
                '''
                log_probs = torch.log(probs)
                log_probs_all = log_probs.detach().clone()

                seq_len = labels.shape[1]
                mp = torch.zeros((bs, seq_len+1)).cuda().scatter_(1, masked_pos_shift.cuda(), torch.ones((bs, seq_len+1)).cuda())
                mp = mp[:, 1:]
                mp_long = mp.long()
                pad_2_zero_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(1))  # -100 -> 0#


                zeros = torch.zeros_like(labels, dtype=torch.long).cuda()
                pads = zeros.clone()
                _, max_ids = torch.max(log_probs, dim=-1)


                y_b = pad_2_zero_labels * (1-mp_long) + max_ids * mp_long
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
                                                                           torch.ones_like(
                                                                               masked_ids).long().cuda())
                    probs = torch.softmax(mask * logits, -1)

                    prob = torch.gather(probs, dim=1, index=masked_ids)
                    masked_ids = masked_ids.reshape(-1, s2, sample_num + 1).transpose(1, 2)
                    prob = prob.reshape(-1, s2, sample_num + 1).transpose(1, 2)
                    log_probs = torch.log(prob)
                    masked_ids = masked_ids.reshape(bs * (sample_num + 1), s2)
                    log_probs = log_probs.reshape(bs * (sample_num + 1), s2)

                    pad_2_zero_labels = pad_2_zero_labels.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    pad_2_zero_labels = pad_2_zero_labels.reshape(bs * (sample_num + 1), -1)

                    mp_long = mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp_long = mp_long.reshape(bs * (sample_num + 1), -1)
                    mp = mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp = mp.reshape(bs * (sample_num + 1), -1)

                    pads = pads.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    pads = pads.reshape(bs * (sample_num + 1), -1)

                y_s = pad_2_zero_labels * (1 - mp_long) + masked_ids * mp_long

                y_zero_s = pads * (1 - mp_long) + masked_ids * mp_long
                y_zero_labels = pads * (1 - mp_long) + pad_2_zero_labels * mp_long

                if self.truth_log_probs:
                    truth_log_probs = log_probs.clone().gather(2, pad_2_zero_labels.unsqueeze(2)).squeeze()
                else:
                    truth_log_probs = None
                # print(log_probs.shape, masked_ids.shape)
                if sample_num != 0:
                    log_probs = log_probs * mp.float()

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.reshape(-1, lm_logits.size(-1)), labels.reshape(-1))

            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
                return ((loss,) + output) if loss is not None else output
            if labels is None:
                return Seq2SeqLMOutput(
                    loss=loss,
                    logits=lm_logits,
                    past_key_values=decoder_outputs.past_key_values,
                    decoder_hidden_states=decoder_outputs.hidden_states,
                    decoder_attentions=decoder_outputs.attentions,
                    cross_attentions=decoder_outputs.cross_attentions,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions,
                )

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            ), y_b, y_s, max_ids, masked_ids, input_ids, labels, \
                non_zero_sum_tensor, log_probs, y_zero_b, y_zero_s, \
                y_zero_labels, truth_log_probs, log_probs_all, lm_logits, \
                mask_labels, masked_pos_shift, masked_pos_non_shift, \
                decoder_input_ids, None, None, None, None, predict_baseline


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


            res = [construct_return(lm_logits=ce_lm_logits, labels=ce_labels, bs=bs // 2,
                                    masked_pos_shift=ce_masked_pos_shift,
                                    masked_pos_non_shift=ce_masked_pos_non_shift, ce=True),
                   construct_return(lm_logits=lm_logits, labels=labels, bs=bs // 2,
                                    masked_pos_shift=masked_pos_shift,
                                    masked_pos_non_shift=masked_pos_non_shift),

                   ]
        else:
            res = construct_return(lm_logits=lm_logits, labels=labels, bs=bs,
                                   masked_pos_shift=masked_pos_shift, masked_pos_non_shift=masked_pos_non_shift)
        return res


    def forward(self, **kwargs):
        if not self.training and "past_key_values" not in kwargs and 'not_seq_decode' not in kwargs:
            bs = kwargs["input_ids"].shape[0]
            with torch.no_grad():
                if self.config.decoding_method == 'non_seq':
                    # return_dict = super().forward(**kwargs)
                    return_dict = self.rl_forward(**kwargs)
                    y_b = return_dict[1]
                    res = y_b
                    neg_mask = ~res.data.eq(-100)
                    res = res * neg_mask
                    res = torch.cat([torch.zeros(bs, 1, dtype=torch.long).cuda(), res], -1)
                   # print(res)



                elif self.config.decoding_method == 'seq':
                    res = self.generate(
                        input_ids = kwargs["input_ids"],
                        attention_mask = kwargs["attention_mask"]
                    )


                #print(res)
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
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
           # return_dict = super().forward(**kwargs)
            return_dict = self.rl_forward(**kwargs)
            return return_dict
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5Model

class CustomT5Model(T5Model):
    def __init__(self, config: T5Config):
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

            >>> from transformers import T5Tokenizer, T5Model

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5Model.from_pretrained('t5-small')

            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

            >>> # forward pass
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        # Decode
        if self.is_scoring_mode:
            cand_num = decoder_input_ids.size(1)
            hidden_states = encoder_outputs[0]
            hidden_states = torch.repeat_interleave(hidden_states, cand_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
            decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            # print(decoder_attention_mask)
        else:
            hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
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

