import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from . import register_model
from config.decorator import replace
from transformers.models.bart.modeling_bart import *
from transformers.file_utils import ModelOutput

from torch.distributions import Categorical
import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from models.mask_policy_utils import MaskPolicy



'''
sharing encoder trick
'''
class CustomBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.is_sharing_encoder = False


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
        if self.is_sharing_encoder:
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
            self.model_parallel = False
            self.device_map = None
            self.vocab_size = config.vocab_size
            self.mask_id = 50264

        except AttributeError as e:
            print(e)
            pass
        self.mask_policy = MaskPolicy(mask_id=self.mask_id, config=config, bpe_prefix='Ġ')

    def forward (self,**kwargs):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            pos = kwargs["input_ids"] == self.mask_id
            e_id = 50118
            # only for samsum, replace [mask] with '\n' in the src.
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
        else:
            if 'data_gt' in kwargs:
                kwargs.pop('data_gt')
            if 'data_src' in kwargs:
                kwargs.pop('data_src')
            if 'data_kd' in kwargs:
                kwargs.pop('data_kd')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')

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

    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bs = None
        mask_labels, masked_pos_shift, masked_pos_non_shift = None, None, None
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            bs = labels.shape[0]
            mask_labels, masked_pos_shift, masked_pos_non_shift = self.mask_policy.get_masked_input(labels, input_ids.shape[0], tokenizer=self.tokenizer)
            decoder_input_ids = shift_tokens_right(
                mask_labels, self.config.pad_token_id, 0
            )
        decoder_input_ids = decoder_input_ids.cuda()

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
        res = self.mask_policy.split_return(lm_logits, input_ids, labels,masked_pos_shift, masked_pos_non_shift, bs,
                                         return_dict, outputs, mask_labels, decoder_input_ids, self.tokenizer)
        return res

    def sharing_encoder(self, flag):
        self.model.is_sharing_encoder = flag