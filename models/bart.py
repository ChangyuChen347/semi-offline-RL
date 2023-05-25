import torch
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from . import register_model
from config.decorator import replace
from transformers.models.bart.modeling_bart import *
from transformers.file_utils import ModelOutput
from models.loss import *
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from numpy import random as np_rand
from torch.distributions import Categorical
import torch.nn as nn
from transformers.models.bart.configuration_bart import BartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
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

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, multiplier=1):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, multiplier, tgt_len, src_len).reshape(-1,1,tgt_len,src_len).to(dtype)

    #print(expanded_mask.size())
    #print(mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype).size())

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


@replace(BartAttention)
class BartAttentionV2(BartAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        kv_bsz = bsz

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
            kv_bsz = key_states.size(0)
        elif is_cross_attention:
            # cross_attentions
            kv_bsz = key_value_states.size(0)
            key_states = self._shape(self.k_proj(key_value_states), -1, kv_bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, kv_bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        kv_proj_shape = (kv_bsz * self.num_heads, -1, self.head_dim)
        key_states = key_states.view(*kv_proj_shape)
        value_states = value_states.view(*kv_proj_shape)

        src_len = key_states.size(1)

        if is_cross_attention:# and kv_bsz != bsz:
            attn_weights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                query_states.view(kv_bsz, -1, self.num_heads, *query_states.size()[1:]),
                key_states.view(kv_bsz, self.num_heads, *key_states.size()[1:]),
            )
            attn_weights = attn_weights.reshape(-1, *attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        if is_cross_attention: #and bsz != kv_bsz:
            attn_probs = attn_probs.to(value_states.dtype)
            attn_output = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attn_probs.view(kv_bsz, -1, self.num_heads, *attn_probs.size()[1:]),
                value_states.view(kv_bsz, self.num_heads, *value_states.size()[1:]),
            )
            attn_output = attn_output.reshape(-1, *attn_output.size()[-2:])
        else:
            attn_probs = attn_probs.to(value_states.dtype)
            attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

@replace(BartDecoder)
class BartDecoder(BartDecoder):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.
                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.
                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            cross_attn_head_mask (:obj:`torch.Tensor` of shape :obj:`(decoder_layers, decoder_attention_heads)`, `optional`):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in ``[0, 1]``:
                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.
            past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
                Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2
                tensors of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional
                tensors of shape :obj:`(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see :obj:`past_key_values` input) to speed up sequential
                decoding.
                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1], multiplier = inputs_embeds.size(0)//encoder_attention_mask.size(0))

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
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
        # ed = time.time() - st
        # print('e', ed)
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
            # decoder_input_ids = decoder_input_ids[:, 1:]
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
            else:
                cand_mask = decoder_input_ids != 1
                cand_mask[:, 0] = 1
                decoder_attention_mask = cand_mask.view(-1, cand_mask.size(-1))
        else:
            # decoder_input_ids = decoder_input_ids[:, 1:]
            encoder_hidden_states = encoder_outputs[0]
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # print(input_ids)
        # print(encoder_outputs)
        # print(decoder_input_ids[:, :10])
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
        # ed = time.time() - st
        # print('d', ed)
        # print(decoder_outputs.last_hidden_state.shape)
        # print('decoder_outputs', decoder_outputs.last_hidden_state[:, :5, :5])
        # print()
        if not return_dict:
            return decoder_outputs + encoder_outputs
        # print(decoder_outputs.last_hidden_state)
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
        # print(res.keys())
        # print(res)
        # print('ed', time.time()-ste)
        return res


class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        # self.summary = Identity()
        # if hasattr(config, "summary_use_proj") and config.summary_use_proj:
        #     if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
        #         num_classes = config.num_labels
        #     else:
        #         num_classes = config.hidden_size
        num_classes = 1
        self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        # if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
        #     self.activation = nn.Tanh()

        # self.first_dropout = Identity()
        # if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
        #     self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.first_dropout = nn.Dropout(0.1)
        self.last_dropout = Identity()
        # if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
        #     self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


@replace(BartForConditionalGeneration)
class BartForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # print(self.model.shared.num_embeddings)
        self.init_weights()
        self.config = config
        if self.config.ppo:
            self.v_head = ValueHead(config)
        try:
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

            self.vocab_size = config.vocab_size
            self.use_cont_mask_id = config.use_cont_mask_id
            self.mask_id = 50264
            self.keep_prob = config.keep_prob
            self.random_prob = config.random_prob
            #self.not_mask_stop = config.not_mask_stop
            self.sample_topk = config.sample_topk
            # print(self.model.shared.num_embeddings)
            if self.config.lm_head2:
                self.lm_head2 = nn.Linear(config.d_model, self.model.shared.num_embeddings+1, bias=True)
            self.is_debug = False
        except AttributeError as e:
            print(e)
            self.is_debug=True
            pass
        # self.value_linear1 =  nn.Linear(config.d_model, config.d_model // 4, bias=True)
        # self.value_linear2 = nn.Linear(config.d_model // 4, 1, bias=True)
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
                # print(self.mask_id+cont)
                if cont > 20:
                    cont = 20
                return self.mask_id + cont

    def forward (self,**kwargs):
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            # print(kwargs["input_ids"])
            # print(kwargs["input_ids"].shape)
            pos = kwargs["input_ids"] == self.mask_id
            # print(pos.shape)
            # e_id = 50260
            e_id = 50118
            kwargs["input_ids"] = pos * torch.ones_like(kwargs["input_ids"]).long().cuda() * e_id + ~pos * kwargs["input_ids"]
            # print(kwargs["input_ids"][0])

        # print(not self.training)
        # print(kwargs)
        # print("past_key_values" not in kwargs)
        # print('not_seq_decode' not in kwargs)
        # assert 1==0
        # print(self.model.is_scoring_mode)
        if not self.training and "past_key_values" not in kwargs and 'not_seq_decode' not in kwargs:
            bs = kwargs["input_ids"].shape[0]
            # tmp_model = super(BartForConditionalGeneration, self).from_pretrained('lmqg/bart-large-squad')
            # self.load_state_dict(tmp_model.state_dict())
            # print(self.tmp_model.model.encoder.embed_tokens.weight[0])
            # print(self.model.encoder.embed_tokens.weight[0])
            # x = [[0, 50265, 12674, 1755, 1437, 50265, 617, 4939, 69, 3501, 756, 6, 8996, 25, 15629, 3250, 381, 16597,
            #       957, 11, 5, 2266, 4388, 4003, 18137, 6, 23906, 10023, 4, 2]]
            # q = self.generate(torch.tensor(x).cuda())
            # print(q)
            # self.load_state_dict(self.tmp_model.state_dict(), False)
            # q = self.generate(torch.tensor(x).cuda())
            # print(q)
            res = self.generate(
                input_ids = kwargs["input_ids"],
                attention_mask = kwargs["attention_mask"],
            )
            # print(res) #bs*beam, seq_len
            # q = self.tmp_model.generate(kwargs["input_ids"].cpu())
            # print(q)
            pad = res.new_full(list(res.shape[:-1])+[max(self.config.max_length - res.shape[-1],0)],0)
            res = torch.cat([res,pad],dim=-1)
            res = res.view([bs,-1] + list(res.shape[1:]))
            # assert 1==0
            #print(res.shape)
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
            # ed = time.time()-st
            # print('all', ed)
            return res
            # return super(BartForConditionalGeneration, self).forward(**kwargs)
        elif 'normal_forward_v' in kwargs:
            kwargs.pop('normal_forward_v')
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
            return self.normal_forward_v(**kwargs)
        elif 'normal_forward_no_mask' in kwargs:
            kwargs.pop('normal_forward_no_mask')
            if 'doc' in kwargs:
                kwargs.pop('doc')  # todo
            if 'query' in kwargs:
                kwargs.pop('query')  # todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            # return super().forward(**kwargs)
            return self.rl_forward_no_mask(**kwargs)
        elif 'normal_forward_second' in kwargs:
            kwargs.pop('normal_forward_second')
            if 'doc' in kwargs:
                kwargs.pop('doc')  # todo
            if 'query' in kwargs:
                kwargs.pop('query')  # todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            # return super().forward(**kwargs)
            return self.rl_forward_second(**kwargs)
        elif 'rl_forward_second_s' in kwargs:
            kwargs.pop('rl_forward_second_s')
            if 'doc' in kwargs:
                kwargs.pop('doc')  # todo
            if 'query' in kwargs:
                kwargs.pop('query')  # todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            # return super().forward(**kwargs)
            return self.rl_forward_second_s(**kwargs)
        elif not self.is_debug:
            if 'doc' in kwargs:
                kwargs.pop('doc') #todo
            if 'query' in kwargs:
                kwargs.pop('query') #todo
            if 'q2' in kwargs:
                kwargs.pop('q2')
            if 'not_seq_decode' in kwargs:
                kwargs.pop('not_seq_decode')
            #return super().forward(**kwargs)
            import time
            st = time.time()
            res = self.rl_forward(**kwargs)
            # ed = time.time() - st
            # print('all', ed)
            return res
        else:
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
            return super(BartForConditionalGeneration, self).forward(**kwargs)


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
        mask_decoder_inputs = True
        # masked_pos_non_shift = None
        # mask_id = 50264
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
                # 1 eos 0 pad -100 ce pad
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
                        intervals = [0]
                        for _k in range(non_zero_sum[i]):
                            if (tmp_tks[i][_k] == ',' or tmp_tks[i][_k] == '.' or tmp_tks[i][_k] == 'Ġ,' or
                                    tmp_tks[i][_k] == 'Ġ.' or _k == non_zero_sum[i]-1):
                                intervals.append(_k)

                        interval_id = random.randint(0, len(intervals)-2)
                        st = intervals[interval_id]
                        ed = intervals[interval_id+1]
                        # print(st, ed)
                        interval_mask = False
                        while k < non_zero_sum[i] - self.tail_unmask_num:
                            if label_np[k] == 0:
                                k += 1
                                continue


                            if tmp_tks[i][k][0] != 'Ġ' and tmp_tks[i][k] != '.' and tmp_tks[i][k] != ',':  # if pre is mask this is not mask it will connect
                                should_mask_pos[i][k] = 1
                            else:
                                should_mask_pos[i][k] = 0

                            '''
                            
                            '''

                            if self.config.v2_0401:
                                if self.config.cand_pos_remove_sp_tk and spmask:
                                    # . x  .后面的词不mask
                                    if self.config.nmask_comma:
                                        if (tmp_tks[i][k] == ',' or tmp_tks[i][k] == '.' or tmp_tks[i][k] == 'Ġ,' or
                                                tmp_tks[i][k] == 'Ġ.'):
                                            k += 1
                                            continue
                                    if self.config.nmask_next_comma:
                                        if k > 0 and (
                                                tmp_tks[i][k - 1] == ',' or tmp_tks[i][k - 1] == '.' or tmp_tks[i][
                                            k - 1] == 'Ġ,' or tmp_tks[i][k - 1] == 'Ġ.'):
                                            k += 1
                                            continue
                            else:
                                if self.config.cand_pos_remove_sp_tk:
                                    # . x  .后面的词不mask
                                    if self.config.nmask_comma:
                                        if (tmp_tks[i][k] == ',' or tmp_tks[i][k] == '.' or tmp_tks[i][k] == 'Ġ,' or
                                                tmp_tks[i][k] == 'Ġ.'):
                                            k += 1
                                            continue
                                    if self.config.nmask_next_comma:
                                        if k > 0 and (tmp_tks[i][k - 1] == ',' or tmp_tks[i][k - 1] == '.' or tmp_tks[i][
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
                        #sample_pos = sorted(sample_pos)
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

                        if self.io_not_same_mask:


                            cont = 0
                            for idx, j in enumerate(sample_pos):
                                if idx > 0 and sample_pos[idx - 1] + 1 == j:
                                    cont += 1
                                else:
                                    cont = 0
                                if self.mask_input:
                                    mask_labels[i][j] = self.get_masked_token(mask_labels[i][j], cont)

                            out_sample_num = int(len(cand_pos) * self.out_mask_rate)
                            sample_pos2 = np_rand.choice(a=np.array(cand_pos), size=out_sample_num,
                                                             replace=False).tolist()
                            sample_pos2 = sorted(sample_pos2)
                            for idx, j in enumerate(sample_pos2):
                                masked_pos_shift[i][idx] = j + 1
                                masked_pos_non_shift[i][idx] = j
                            for idx, j in enumerate(non_sample_pos):
                                if random.random() <= 1:
                                    non_masked_pos_shift[i][idx] = j

                        else:
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
                #print(decoder_input_ids)
                #print(decoder_input_ids)
                #assert 1==0
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

            if mask_decoder_inputs and masked_pos_non_shift is not None:
                topk = self.sample_topk
                if topk == -1:
                    probs = torch.softmax(lm_logits, dim=-1)
                    if self.config.sample_method != 'multi':
                        to_sample_lm_logits = lm_logits.clone().detach()
                        to_sample_probs = torch.softmax(to_sample_lm_logits, dim=-1)

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

                # if self.config.normalize:
                #     log_probs = log_probs / ((mp_long.sum(-1) + 1e-8) ** self.config.normalize_penalty).unsqueeze(1).unsqueeze(1)
                #     log_probs = log_probs / log_probs.shape[0]

                # ori_log_probs = log_probs.clone()
                ori_mp = mp.clone()
                ones = torch.ones_like(labels, dtype=torch.long).cuda()

                other_2_pad_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(2)) + ones * (labels.data.eq(-100) | labels.data.eq(2)) # -100 -> 0#
                ori_other_2_pad_labels = other_2_pad_labels.clone()
                pads = ones.clone()
                _, max_ids = torch.max(log_probs, dim=-1)
                # print(max_ids.shape)
                # print(labels)
                eos_mask = labels == 2

                eos_log_probs = log_probs.gather(2, torch.ones_like(labels).long().cuda().unsqueeze(2)*2).squeeze()
                eos_log_probs = eos_log_probs * eos_mask
                y_b = other_2_pad_labels * (1-mp_long) + max_ids * mp_long
                y_zero_b = pads * (1-mp_long) + max_ids * mp_long

                # non_mp = torch.zeros((bs, seq_len + 1)).cuda().scatter_(1, non_masked_pos_shift,
                #                                                         torch.ones((bs, seq_len + 1)).cuda())
                # non_mp = non_mp[:, 1:]
                # non_mp_long = non_mp.long()
                # y_b_g = other_2_pad_labels * (1 - mp_long) + max_ids * mp_long
                # y_b_g = y_b_g * (1 - non_mp_long) + max_ids * non_mp_long
                # y_b_g_m = y_b_g * (1 - mp_long) + torch.ones_like(max_ids).long().cuda() * self.mask_id * mp_long
                sample_num = 0
                if not ce:
                    sample_num = self.config.sample_num
                if sample_num != 0:
                    _, s2, s3 = probs.shape

                    probs = probs.reshape(-1, s3)
                    if self.config.use_logit:
                        logits = lm_logits.reshape(-1, s3)

                    masked_ids = torch.multinomial(probs, sample_num + 1, replacement=True)
                    # bs * seq_len, sample_num
                    if self.config.use_logit:
                        #logit = torch.gather(logit, dim=1, index=masked_ids) #bs, seq_len, sample_num
                        # mask = torch.zeros_like(logit).cuda().long().scatter_(1, masked_ids,torch.ones_like(masked_ids).long().cuda())

                        mask = torch.zeros_like(logits).cuda().long().scatter_(1, masked_ids,
                                                       torch.ones_like(masked_ids).long().cuda())
                        # print(probs[0])

                        probs = torch.softmax(mask*logits, -1)

                        # print(probs[0])
                    ori_masked_ids = masked_ids
                    prob = torch.gather(probs, dim=1, index=masked_ids)
                    if self.config.prob_w:
                        prob_w = torch.gather(probs.detach().clone(), dim=1, index=masked_ids)

                    masked_ids = masked_ids.reshape(-1, s2, sample_num + 1).transpose(1,2)
                    if self.config.prob_w:
                        prob_w = prob_w.reshape(-1, s2, sample_num + 1).transpose(1,2)
                    prob = prob.reshape(-1, s2, sample_num + 1).transpose(1,2)
                    log_probs = torch.log(prob)
                    # if self.config.prob_w:
                    #     log_probs = log_probs * prob_w
                    masked_ids = masked_ids.reshape(bs * (sample_num + 1), s2)
                    log_probs = log_probs.reshape(bs * (sample_num + 1), s2)

                    other_2_pad_labels = other_2_pad_labels.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    other_2_pad_labels = other_2_pad_labels.reshape(bs * (sample_num + 1), -1)
                    mp_long = mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp_long = mp_long.reshape(bs * (sample_num + 1), -1)
                    mp = mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    mp = mp.reshape(bs * (sample_num + 1), -1)

                    # non_mp_long = non_mp_long.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    # non_mp_long = non_mp_long.reshape(bs * (sample_num + 1), -1)
                    # non_mp = non_mp.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    # non_mp = non_mp.reshape(bs * (sample_num + 1), -1)
                    pads = pads.unsqueeze(1).expand(-1, sample_num + 1, -1)
                    pads = pads.reshape(bs * (sample_num + 1), -1)
                else:
                    _, s2, s3 = probs.shape
                    probs = probs.reshape(-1, s3)
                    if self.config.use_logit:
                        logits = lm_logits.reshape(-1, s3)

                    masked_ids = torch.multinomial(probs, sample_num + 1, replacement=True)
                    # bs * seq_len, sample_num
                    if self.config.use_logit:
                        # logit = torch.gather(logit, dim=1, index=masked_ids) #bs, seq_len, sample_num
                        # mask = torch.zeros_like(logit).cuda().long().scatter_(1, masked_ids,torch.ones_like(masked_ids).long().cuda())

                        mask = torch.zeros_like(logits).cuda().long().scatter_(1, masked_ids,
                                                                               torch.ones_like(masked_ids).long().cuda())
                        # print(probs[0])

                        probs = torch.softmax(mask * logits, -1)

                        # print(probs[0])
                    prob = torch.gather(probs, dim=1, index=masked_ids)
                    if self.config.prob_w:
                        prob_w = torch.gather(probs.detach().clone(), dim=1, index=masked_ids)

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


                if self.truth_log_probs:
                    pass
                    # truth_log_probs = ori_log_probs.clone().gather(2, ori_other_2_pad_labels.unsqueeze(2)).squeeze()
                else:
                    truth_log_probs = None
                #if self.greedy_log_probs:
                #greedy_log_probs = ori_log_probs.clone().gather(2, max_ids.unsqueeze(2)).squeeze()
                #greedy_log_probs = greedy_log_probs * ori_mp.float()
                if truth_log_probs is not None:
                    truth_log_probs = truth_log_probs * ori_mp.float()

                if sample_num != 0 and self.config.sample_method == 'multi':

                    log_probs = log_probs * mp.float()
                else:
                    if self.config.use_all_probs:
                        masked_probs = log_probs.gather(2, y_s.unsqueeze(2)).squeeze()
                        mp_all = ~(labels.data.eq(-100) | labels.data.eq(0) | labels.data.eq(2))
                        mp_all = mp_all.unsqueeze(1).expand(-1, sample_num + 1, -1)
                        mp_all = mp_all.reshape(bs * (sample_num + 1), -1)
                        log_probs = masked_probs * mp_all.float()
                    else:
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
                       log_probs, y_zero_b, y_zero_s, y_zero_labels, truth_log_probs, log_probs_all, lm_logits, \
                       mask_labels, masked_pos_shift, \
                       masked_pos_non_shift, decoder_input_ids, None, None, None, None, None, eos_log_probs
        st = time.time()
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
        # ed = time.time()-st
        # print('cons', ed)
        return res
    def normal_forward_v(
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
            if decoder_input_ids is None and decoder_inputs_embeds is None:
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
        value = self.v_head(outputs[0]).squeeze(-1)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        # print(return_dict)
        if not return_dict:
            output = (lm_logits,) + outputs[1:2] + (value,)
            # outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
            # print(len(output))
            return output

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
            value=value
        )

    def rl_forward_second(
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
            ori_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mask_decoder_inputs = True
        masked_pos_non_shift = None
        mask_id = 50264
        if labels is not None:
            bs = labels.shape[0]
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # x = self.tokenizer.encode('<mask>')
                # print(labels )
                # decoder_input_ids = shift_tokens_right(
                #    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )
                # print(decoder_input_ids)

                bs = labels.shape[0]
                mask_labels = labels.clone()  # bs, seq
                if masked_pos_shift is None:
                    masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                    masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
                    non_zero_labels = ~(
                            labels.data.eq(1) | labels.data.eq(2) | labels.data.eq(-100))  # 0 pad 2 eos -100 pad
                    # 1 eos 0 pad -100 ce pad
                    non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
                    non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()
                    labels_numpy = mask_labels.detach().cpu().numpy().tolist()
                    for i in range(bs):
                        label_np = labels_numpy[i]
                        cand_pos = []
                        k = 0

                        while k < non_zero_sum[i] - self.tail_unmask_num:
                            if label_np[k] == 0:
                                k += 1
                                continue
                            cand_pos.append(k)
                            k += 1
                        for idx, j in enumerate(cand_pos):
                            masked_pos_shift[i][idx] = j + 1
                            masked_pos_non_shift[i][idx] = j

                # decoder_input_ids = shift_tokens_right(
                #     mask_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )  # 2  +labels + 1 pad
                decoder_input_ids = shift_tokens_right(
                    mask_labels, self.config.pad_token_id, 0
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
        probs = torch.softmax(lm_logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        seq_len = labels.shape[1]

        mp = torch.zeros((bs, seq_len + 1)).cuda().scatter_(1, masked_pos_shift,
                                                       torch.ones((bs, seq_len + 1)).cuda())
        mp = mp[:, 1:]
        ones = torch.ones_like(labels, dtype=torch.long).cuda()
        other_2_pad_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(2)) + ones * (
                    labels.data.eq(-100) | labels.data.eq(2))  # -100 -> 0#

        masked_probs = log_probs.gather(2, other_2_pad_labels.unsqueeze(2)).squeeze()
        log_probs = masked_probs * mp.float()
        masked_lm_loss = None

        if ori_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), ori_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # print(labels is None)
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
            ), log_probs

    def rl_forward_second_s(
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
            ori_labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        mask_decoder_inputs = True
        masked_pos_non_shift = None
        mask_id = 50264
        if labels is not None:
            bs = labels.shape[0]
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                # x = self.tokenizer.encode('<mask>')
                # print(labels )
                # decoder_input_ids = shift_tokens_right(
                #    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )
                # print(decoder_input_ids)

                bs = labels.shape[0]
                mask_labels = labels.clone()  # bs, seq
                non_zero_labels = ~(
                        labels.data.eq(1) | labels.data.eq(2) | labels.data.eq(-100))  # 0 pad 2 eos -100 pad
                # 1 eos 0 pad -100 ce pad
                non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
                non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()

                if masked_pos_shift is None:
                    masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                    masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq

                    labels_numpy = mask_labels.detach().cpu().numpy().tolist()
                    for i in range(bs):
                        label_np = labels_numpy[i]
                        cand_pos = []
                        k = 0
                        while k < non_zero_sum[i] - self.tail_unmask_num:
                            if label_np[k] == 0:
                                k += 1
                                continue
                            cand_pos.append(k)
                            k += 1
                        for idx, j in enumerate(cand_pos):
                            masked_pos_shift[i][idx] = j + 1
                            masked_pos_non_shift[i][idx] = j

                # decoder_input_ids = shift_tokens_right(
                #     mask_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )  # 2  +labels + 1 pad
                decoder_input_ids = shift_tokens_right(
                    mask_labels, self.config.pad_token_id, 0
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
        topk = self.sample_topk
        if topk == -1:
            to_sample_lm_logits = lm_logits
            to_sample_probs = torch.softmax(to_sample_lm_logits, dim=-1)
            probs = to_sample_probs
        else:
            indices_to_remove = lm_logits < torch.topk(lm_logits, topk)[0][..., -1, None]
            indices_to_keep = lm_logits >= torch.topk(lm_logits, topk)[0][..., -1, None]
            # print(indices_to_remove.shape)
            # print(probs.shape)
            to_sample_lm_logits = lm_logits * indices_to_keep.cuda() + indices_to_remove.cuda() * torch.ones_like(
                lm_logits).cuda() * -1e8
            to_sample_probs = torch.softmax(to_sample_lm_logits, dim=-1)
            probs = torch.softmax(lm_logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        # seq_len = labels.shape[1]
        # mp = torch.zeros((bs, seq_len + 1)).cuda().scatter_(1, masked_pos_shift,
        #                                                torch.ones((bs, seq_len + 1)).cuda())
        # mp = mp[:, 1:]
        # ones = torch.ones_like(labels, dtype=torch.long).cuda()
        # other_2_pad_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(2) | labels.data.eq(0)) + ones * (
        #             labels.data.eq(-100) | labels.data.eq(2) | labels.data.eq(0))  # -100 -> 0#
        #
        # masked_probs = log_probs.gather(2, other_2_pad_labels.unsqueeze(2)).squeeze()
        # log_probs = masked_probs * mp.float()
        '''
        '''
        seq_len = labels.shape[1]
        mp = torch.zeros((bs, seq_len + 1)).cuda().scatter_(1, masked_pos_shift, torch.ones((bs, seq_len + 1)).cuda())
        mp = mp[:, 1:]
        mp_long = mp.long()
        if self.config.normalize:
            log_probs = log_probs / ((mp_long.sum(-1) + 1e-8) ** self.config.normalize_penalty).unsqueeze(1).unsqueeze(
                1)
            log_probs = log_probs / log_probs.shape[0]
        # ori_log_probs = log_probs.clone()
        ori_mp = mp.clone()
        ones = torch.ones_like(labels, dtype=torch.long).cuda()
        other_2_pad_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(2)) + ones * (
                    labels.data.eq(-100) | labels.data.eq(2))  # -100 -> 0#
        ori_other_2_pad_labels = other_2_pad_labels.clone()
        pads = ones.clone()
        _, max_ids = torch.max(log_probs, dim=-1)
        y_b = other_2_pad_labels * (1 - mp_long) + max_ids * mp_long
        y_zero_b = pads * (1 - mp_long) + max_ids * mp_long
        multi_dist = Categorical(to_sample_probs)
        masked_ids = multi_dist.sample()
        y_s = other_2_pad_labels * (1 - mp_long) + masked_ids * mp_long
        y_zero_s = pads * (1 - mp_long) + masked_ids * mp_long
        y_zero_labels = pads * (1 - mp_long) + other_2_pad_labels * mp_long

        if self.truth_log_probs:
            pass
            # truth_log_probs = ori_log_probs.clone().gather(2, ori_other_2_pad_labels.unsqueeze(2)).squeeze()
        else:
            truth_log_probs = None
        if truth_log_probs is not None:
            truth_log_probs = truth_log_probs * ori_mp.float()
        masked_probs = log_probs.gather(2, masked_ids.unsqueeze(2)).squeeze()
        log_probs_all = log_probs.clone()
        log_probs = masked_probs * mp.float()

        masked_lm_loss = None

        if ori_labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), ori_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        # print(labels is None)
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
                   log_probs, y_zero_b, y_zero_s, y_zero_labels, truth_log_probs, log_probs_all, lm_logits, mask_labels, masked_pos_shift, masked_pos_non_shift, decoder_input_ids


    def rl_forward_no_mask(
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
        mask_decoder_inputs = True
        masked_pos_non_shift = None
        mask_id = 50264
        if labels is not None:
            bs = labels.shape[0]
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                bs = labels.shape[0]
                mask_labels = labels.clone()  # bs, seq
                masked_pos_shift = torch.zeros_like(mask_labels)  # bs, seq
                masked_pos_non_shift = torch.zeros_like(mask_labels)  # bs, seq
                non_zero_labels = ~(
                        labels.data.eq(1) | labels.data.eq(2) | labels.data.eq(-100))  # 0 pad 2 eos -100 pad
                # 1 eos 0 pad -100 ce pad
                non_zero_sum_tensor = non_zero_labels.sum(-1)  # bs
                non_zero_sum = non_zero_sum_tensor.detach().cpu().numpy().tolist()
                labels_numpy = mask_labels.detach().cpu().numpy().tolist()
                for i in range(bs):
                    label_np = labels_numpy[i]
                    cand_pos = []
                    k = 0
                    while k < non_zero_sum[i] - self.tail_unmask_num:
                        if label_np[k] == 0:
                            k += 1
                            continue
                        cand_pos.append(k)
                        k += 1
                    sample_num = int(len(cand_pos) * self.mask_rate)
                    sample_pos = np_rand.choice(a=np.array(cand_pos), size=sample_num, replace=False).tolist()
                    sample_pos = sorted(sample_pos)
                    for idx, j in enumerate(sample_pos):
                        masked_pos_shift[i][idx] = j + 1
                        masked_pos_non_shift[i][idx] = j
                # decoder_input_ids = shift_tokens_right(
                #     mask_labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )  # 2  +labels + 1 pad
                decoder_input_ids = shift_tokens_right(
                    mask_labels, self.config.pad_token_id, 0
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


        if mask_decoder_inputs and masked_pos_non_shift is not None:
            probs = torch.softmax(lm_logits, dim=-1)
            log_probs = torch.log(probs+ 1e-8)
            seq_len = labels.shape[1]
            mp = torch.zeros((bs, seq_len+1)).cuda().scatter_(1, masked_pos_shift, torch.ones((bs, seq_len+1)).cuda())
            mp = mp[:, 1:]
            mp_long = mp.long()
            if self.config.normalize:
                log_probs = log_probs / ((mp_long.sum(-1) + 1e-8) ** self.config.normalize_penalty).unsqueeze(1).unsqueeze(1)
                log_probs = log_probs / log_probs.shape[0]
            ori_log_probs = log_probs.clone()
            ori_mp = mp.clone()
            ones = torch.ones_like(labels, dtype=torch.long).cuda()
            other_2_pad_labels = labels * ~(labels.data.eq(-100) | labels.data.eq(2) | labels.data.eq(0)) + ones * (labels.data.eq(-100) | labels.data.eq(2) | labels.data.eq(0)) # -100 -> 0#
            ori_other_2_pad_labels = other_2_pad_labels.clone()
            pads = ones.clone()
            _, max_ids = torch.max(log_probs, dim=-1)

            y_b = other_2_pad_labels * (1-mp_long) + max_ids * mp_long
            y_zero_b = pads * (1-mp_long) + max_ids * mp_long

            multi_dist = Categorical(probs)
            masked_ids = multi_dist.sample()
            y_s = other_2_pad_labels * (1 - mp_long) + masked_ids * mp_long
            y_zero_s = pads * (1 - mp_long) + masked_ids * mp_long
            y_zero_labels = pads * (1 - mp_long) + other_2_pad_labels * mp_long

            if self.truth_log_probs:
                truth_log_probs = ori_log_probs.clone().gather(2, ori_other_2_pad_labels.unsqueeze(2)).squeeze()
            else:
                truth_log_probs = None
            #if self.greedy_log_probs:
            #greedy_log_probs = ori_log_probs.clone().gather(2, max_ids.unsqueeze(2)).squeeze()
            #greedy_log_probs = greedy_log_probs * ori_mp.float()

            if truth_log_probs is not None:
                truth_log_probs = truth_log_probs * ori_mp.float()
            # print(masked_ids.shape)
            # print(log_probs.shape)
            masked_probs = log_probs.gather(2, masked_ids.unsqueeze(2)).squeeze()
            # print(masked_probs.shape)
            # print(mp.shape)
            log_probs_all = log_probs.clone()
            log_probs = masked_probs * mp.float()

        masked_lm_loss = None


        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

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
                   log_probs, y_zero_b, y_zero_s, y_zero_labels, truth_log_probs, log_probs_all, lm_logits


    @staticmethod
    def _expand_inputs_for_generation(
        #self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        #if "token_type_ids" in model_kwargs:
        #    token_type_ids = model_kwargs["token_type_ids"]
        #    model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)
        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state
        #    encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
        #        0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
        #    )
            model_kwargs["encoder_outputs"] = encoder_outputs
            
        return input_ids, model_kwargs

@register_model("guided_seq2seq","Bart")
class BartForLatentGuidedGeneration(BartForConditionalGeneration):
    set_cfg=True
    def __init__(self,config,customize_cfg):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.customize_cfg = customize_cfg
        self.latent_class = self.customize_cfg.latent_class
        self.selected_class = self.customize_cfg.selected_class
        latent_dict = [288,134,176,246,306,245,401,406,398,466,698]
        self.latent_signal = torch.tensor(latent_dict[:self.latent_class],dtype=torch.int32)
        self.cls_layer = BartEncoderLayer(config)
        self.enc_cls_layer = BartEncoderLayer(config)
        self.cls_label = torch.arange(0,self.latent_class)
        self.classifier = BartClassificationHead(
                            config.d_model,
                            config.d_model,
                            self.latent_class,
                            config.classifier_dropout)
        self.enc_classifier = BartClassificationHead(
                                config.d_model,
                                config.d_model,
                                self.latent_class,
                                config.classifier_dropout)
        self.embed_position = BartLearnedPositionalEmbedding(
                            config.max_position_embeddings,
                            config.d_model)
        self.cls_loss_weight = self.customize_cfg.cls_loss_weight
        self.loss_fct = LatentGuidedLoss(self.latent_class, self.cls_loss_weight)

        self.init_weights()


    def latent_cls(self,ids,mask,cls_layer,classifier):
        mask = torch.where(ids==-100,0,1) if mask is None else mask
        ids = torch.where(ids==-100,0,ids)
        embs = self.model.get_input_embeddings()(ids)
        mask = _expand_mask(mask,embs.dtype)
        embs += self.embed_position(ids.size())
        encoder_output = cls_layer(embs,mask,None)[0]
        logits = classifier(encoder_output[:,0,:])
        return logits

    def forward(
        self,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        if not self.training and "past_key_values" not in kwargs:
            bs = kwargs["input_ids"].shape[0]
            res = self.generate(
                input_ids = kwargs["input_ids"],
                attention_mask = kwargs["attention_mask"]
            )
            res = res[:,3:]
            pad = res.new_full(list(res.shape[:-1])+[max(self.config.max_length - res.shape[-1],0)],0)
            res = torch.cat([res,pad],dim=-1)
            #res = res.view([bs,-1] + list(res.shape[1:]))
            enc_logits = self.latent_cls(kwargs["input_ids"],kwargs["attention_mask"],self.enc_cls_layer,self.enc_classifier)
            selected_cate = torch.topk(enc_logits,self.selected_class,dim=-1).indices.flatten()
            selected_cate += torch.arange(0,bs,dtype=selected_cate.dtype).to(selected_cate.device).unsqueeze(-1).tile([1,self.selected_class]).flatten()*self.latent_class
            
            res = res.index_select(0,selected_cate)
            res = res.view([bs,-1] + list(res.shape[1:]))
            return {"loss":pad.new_full([1],0.0,dtype=torch.float32),"ids":res}
        #NOT BEAM SEARCH
        
        input_ids=kwargs.get("input_ids",None)
        attention_mask=kwargs.get("attention_mask",None)
        decoder_input_ids = kwargs.get("decoder_input_ids",None)
        decoder_attention_mask = kwargs.get("decoder_attention_mask",None)
        head_mask=kwargs.get("head_mask",None)
        decoder_head_mask = kwargs.get("decoder_head_mask",None)
        cross_attn_head_mask=kwargs.get("cross_attn_head_mask",None)
        encoder_outputs = kwargs.get("encoder_outputs",None)
        past_key_values=kwargs.get("past_key_values",None)
        inputs_embeds=kwargs.get("inputs_embeds",None)
        decoder_inputs_embeds=kwargs.get("decoder_inputs_embeds",None)
        labels=kwargs.get("labels",None)
        use_cache=kwargs.get("use_cache",None)
        output_attentions=kwargs.get("output_attentions",None)
        output_hidden_states=kwargs.get("output_hidden_states",None)
        return_dict=kwargs.get("return_dict",None)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            bsz,seqlen = labels.size()
            #Calculate latent class, add latent signal 
            latent_signal = self.latent_signal.to(labels.device).tile([bsz]).unsqueeze(-1)
            latent_logits = self.latent_cls(labels,None,self.cls_layer,self.classifier)
            enc_logits = self.latent_cls(input_ids,attention_mask,self.enc_cls_layer,self.enc_classifier)

            labels = labels.tile([1,self.latent_class]).view(bsz*self.latent_class,-1)
            labels = torch.cat([labels[:,:1],latent_signal,labels[:,1:]],dim=-1)
            if decoder_input_ids is None:
                # decoder_input_ids = shift_tokens_right(
                #     labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, 0
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
        if self.config.lm_head2:
            lm_logits2 = self.lm_head2(outputs[0])

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None

        mask_decoder_inputs = True
        masked_pos_non_shift = None
        #if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:

        if labels is not None:
            masked_lm_loss = self.loss_fct(lm_logits[:,2:,:],labels[:,2:],latent_logits,enc_logits)
                
        ###lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

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
    
    #@staticmethod
    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        input_ids = input_ids.tile([expand_size,1])
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state
            model_kwargs["encoder_outputs"] = encoder_outputs
            
        return input_ids, model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((input_ids.shape[0]*self.latent_class, 1), dtype=torch.long, device=input_ids.device) * decoder_start_token_id
        )
        return decoder_input_ids

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            if past[0][0].size()[2] == 2:
                decoder_input_ids = self.latent_signal.to(decoder_input_ids.device).tile(decoder_input_ids.size()[0]//self.latent_class).unsqueeze(-1)
            else:
                decoder_input_ids = decoder_input_ids[:, -1:]
            #print(decoder_input_ids,past[0][0].size())
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

@register_model("selector_seq2seq","Bart")
class BartForSelectorGeneration(BartForConditionalGeneration):
    set_cfg=True
    def __init__(self,config,customize_cfg):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.customize_cfg = customize_cfg
        self.embed_position = BartLearnedPositionalEmbedding(
                            config.max_position_embeddings,
                            config.d_model)
        # sample_pos == 0: sample input_ids before encoder; use sample_encode_layer to encode input_ids
        # sample_pos == 1: sample encoder_outputs after encoder; share encoder-decoder's encoder to encode input_ids
        self.sample_pos = self.customize_cfg.sample_pos
        assert self.sample_pos == 0 or self.sample_pos == 1
        self.sample_num = self.customize_cfg.sample_num
        self.sample_loss_weight = self.customize_cfg.sample_loss_weight
        self.sample_encode_layer = BartEncoderLayer(config)
        self.sample_classifier_layer = BartClassificationHead(
                            config.d_model,
                            config.d_model,
                            1,
                            config.classifier_dropout)

        self.latent_num = self.customize_cfg.latent_num
        latent_dict = [288,134,176,246,306,245,401,406,398,466,698]
        self.latent_signal = torch.tensor(latent_dict[:self.latent_num],dtype=torch.int32)
        # if self.latent_loss_weight == 0, not use latent classifier
        self.latent_loss_weight = self.customize_cfg.latent_loss_weight
        self.latent_encode_layer = BartEncoderLayer(config)
        self.latent_enc_fusion = self.customize_cfg.latent_enc_fusion
        if self.latent_enc_fusion == 'concat':
            input_dim = config.d_model * 2
        elif self.latent_enc_fusion == 'cosine':
            input_dim = 1
        elif self.latent_enc_fusion == 'dot':
            input_dim = config.d_model
        else:
            assert self.latent_enc_fusion == 'concat' or self.latent_enc_fusion == 'cosine' or self.latent_enc_fusion == 'dot'
        self.latent_classifier_layer = BartClassificationHead(
                    input_dim,
                    config.d_model,
                    self.latent_num,
                    config.classifier_dropout)
        if self.latent_loss_weight > 0:
            assert self.latent_num > 1

                            
        self.loss_fct = SelectorGenLoss(self.sample_num, self.sample_loss_weight, self.latent_num, self.latent_loss_weight)

        self.init_weights()

    def latent_classifier(self, input_ids, labels_ids):
        # encoder_outputs (page_num*sample_num)*src_seq_len*d_model
        # labels page_num*tgt_len

        # decoder_outputs (page_num*sample_num*latent_num)*tgt_seq_len*d_model
        # decoder_inputs example: [2,0,latent_variabel,x,x,x,1,1,1,1], we should take the first "token".
        labels_mask = torch.where(labels_ids==-100,0,1)
        labels_ids = torch.where(labels_ids==-100,0,labels_ids)
        labels_embs = self.model.get_input_embeddings()(labels_ids)
        labels_mask = _expand_mask(labels_mask,labels_embs.dtype)
        labels_embs += self.embed_position(labels_ids.size())
        labels_encoder_output = self.latent_encode_layer(labels_embs,labels_mask,None)[0] # page_num*tgt_len*d_model
        page_num, seq_len, d_model = labels_encoder_output.shape
        labels_encoder_output = labels_encoder_output.tile([1,self.sample_num,1]).view(page_num*self.sample_num,seq_len,d_model) # (page_num*sample_num)*tgt_len*d_model
        
        inputs_mask = torch.where(input_ids==0,0,1)
        inputs_embs = self.model.get_input_embeddings()(input_ids)
        inputs_mask = _expand_mask(inputs_mask,inputs_embs.dtype)
        inputs_embs += self.embed_position(input_ids.size())
        inputs_encoder_output = self.latent_encode_layer(inputs_embs,inputs_mask,None)[0] # (page_num*sample_num)*src_len*d_model

        if self.latent_enc_fusion == 'concat':
            logits = self.latent_classifier_layer(torch.cat((inputs_encoder_output[:,0,:], labels_encoder_output[:,0,:]), dim=-1)) # input_dim = 2* d_model
        elif self.latent_enc_fusion == 'cosine':
            inputs_norm = torch.nn.functional.normalize(inputs_encoder_output[:,0,:], dim=1)
            labels_norm = torch.nn.functional.normalize(labels_encoder_output[:,0,:], dim=1)
            logits = self.latent_classifier_layer(torch.sum(inputs_norm*labels_norm, dim=1).unsqueeze(-1)) # input_dim = 1
        elif self.latent_enc_fusion == 'dot':
            inputs_norm = torch.nn.functional.normalize(inputs_encoder_output[:,0,:], dim=1)
            labels_norm = torch.nn.functional.normalize(labels_encoder_output[:,0,:], dim=1) # input_dim = d_model
            logits = self.latent_classifier_layer(inputs_norm * labels_norm)
        else:
            assert self.latent_enc_fusion == 'concat' or self.latent_enc_fusion == 'cosine' or self.latent_enc_fusion == 'dot'

        return logits

    def selector_select(
        self,
        input_ids,
        encoder_outputs,
        attention_mask,
        sentence_num,
        mode
    ):
        # classification
        if self.sample_pos == 0:
            inputs_embs = self.model.get_input_embeddings()(input_ids)
            inputs_mask = _expand_mask(attention_mask, inputs_embs.dtype)
            inputs_embs += self.embed_position(input_ids.size())
            inputs_encoder_output = self.sample_encode_layer(inputs_embs, inputs_mask, None)[0]  # page_num*tgt_len*d_model
        else:
            inputs_encoder_output = encoder_outputs[0]
        select_logits = self.sample_classifier_layer(inputs_encoder_output[:,0,:])
        # select
        ## split and softmax
        select_logits = torch.split(select_logits.squeeze(-1), sentence_num)
        select_logits = pad_sequence(select_logits,batch_first=True,padding_value=float('-inf'))
        select_probs = torch.softmax(select_logits.float(), dim=-1)

        ## sample
        if mode == "sample":
            selected_idxs = torch.multinomial(select_probs, self.sample_num) 
        elif mode == "top":
            _, selected_idxs = torch.topk(select_probs, k=self.sample_num, dim=-1)
        else:
            assert mode == "sample" or mode == "top"
        sentence_num_tensor = torch.tensor(sentence_num, device=select_probs.device).unsqueeze(-1)
        select_mask = (selected_idxs < sentence_num_tensor).float()
        selected_idxs %= sentence_num_tensor
        selected_logits = torch.gather(select_logits, dim=-1, index=selected_idxs)
        selected_probs = torch.gather(select_probs, dim=-1, index=selected_idxs)
        self.selected_logits = torch.sigmoid(selected_logits)

        ## flatten selected_idxs
        page_offset = torch.cumsum(torch.tensor([0] + sentence_num[:-1], device=selected_idxs.device), dim=0).unsqueeze(-1)
        selected_idxs_flatten = selected_idxs + page_offset
        selected_idxs_flatten = torch.flatten(selected_idxs_flatten)     
        
        # select corresponding sentence
        if self.sample_pos == 1:
            encoder_outputs["last_hidden_state"] = torch.index_select(encoder_outputs[0], dim=0, index=selected_idxs_flatten)
        input_ids = torch.index_select(input_ids, dim=0, index=selected_idxs_flatten)
        attention_mask = torch.index_select(attention_mask, dim=0, index=selected_idxs_flatten)

        return input_ids, encoder_outputs, attention_mask, selected_logits, select_mask, selected_idxs, selected_probs

    def forward(
        self,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.
        Returns:
        """
        if not self.training and "past_key_values" not in kwargs:
            input_ids = kwargs["input_ids"]
            sentence_num = [x[0] for x in kwargs["sentence_num"]]
            attention_mask = kwargs["attention_mask"]
            bs = len(sentence_num)
            input_ids = torch.split(input_ids, sentence_num)
            res = self.generate(
                input_ids = input_ids,
                attention_mask = attention_mask
            )
            if self.latent_num > 1:
                res = res[:, 3:]
            pad = res.new_full(list(res.shape[:-1])+[max(self.config.max_length - res.shape[-1],0)],0)
            res = torch.cat([res,pad],dim=-1)
            res = res.view([bs,-1] + list(res.shape[1:]))
            if kwargs.get("labels",None) is None:
                return {"loss":pad.new_full([1],0.0,dtype=torch.float32),"ids":res,"selected_idxs":self.selected_idxs,"selected_probs":self.selected_logits,"latent_num":torch.tensor(self.latent_num)}
            else:
                return {"loss":pad.new_full([1],0.0,dtype=torch.float32),"ids":res}
        #NOT BEAM SEARCH
        
        input_ids=kwargs.get("input_ids",None)
        attention_mask=kwargs.get("attention_mask",None)
        decoder_input_ids = kwargs.get("decoder_input_ids",None)
        decoder_attention_mask = kwargs.get("decoder_attention_mask",None)
        head_mask=kwargs.get("head_mask",None)
        decoder_head_mask = kwargs.get("decoder_head_mask",None)
        cross_attn_head_mask=kwargs.get("cross_attn_head_mask",None)
        encoder_outputs = kwargs.get("encoder_outputs",None)
        past_key_values=kwargs.get("past_key_values",None)
        inputs_embeds=kwargs.get("inputs_embeds",None)
        decoder_inputs_embeds=kwargs.get("decoder_inputs_embeds",None)
        labels=kwargs.get("labels",None)
        use_cache=kwargs.get("use_cache",None)
        output_attentions=kwargs.get("output_attentions",None)
        output_hidden_states=kwargs.get("output_hidden_states",None)
        return_dict=kwargs.get("return_dict",None)
        sentence_num = kwargs.get("sentence_num",None)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        original_labels = labels
        if labels is not None:
            bsz,seqlen = labels.size()
            labels = labels.tile([1,self.latent_num*self.sample_num]).view(bsz*self.latent_num*self.sample_num,-1)
            if self.latent_num > 1:
                latent_signal = self.latent_signal.to(labels.device).tile([bsz*self.sample_num]).unsqueeze(-1) # page_num*latent_num*sample_num
                labels = torch.cat([labels[:,:1],latent_signal,labels[:,1:]],dim=-1)
            if decoder_input_ids is None:
                # decoder_input_ids = shift_tokens_right(
                #     labels, self.config.pad_token_id, self.config.decoder_start_token_id
                # )
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, 0
                )


        # [start] model part
        # 1. encoder part
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        # print(decoder_input_ids)

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.model.config.pad_token_id, self.model.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict

        if encoder_outputs is None:
            sentence_num = [x[0] for x in sentence_num]
            if self.sample_pos == 0:
                input_ids, _, attention_mask, selected_logits, select_mask, selected_idxs, selected_probs = self.selector_select(input_ids, None, attention_mask,sentence_num,"sample")
            encoder_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
            if self.sample_pos == 1:
                input_ids, encoder_outputs, attention_mask, selected_logits, select_mask, selected_idxs, selected_probs = self.selector_select(input_ids, encoder_outputs,attention_mask,sentence_num,"sample")
            latent_logits = None
            if self.latent_num > 1 and self.latent_loss_weight > 0:
                latent_logits = self.latent_classifier(input_ids, original_labels) # (page_num*sample_num)*num_latent_num
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        

        # 3. decoder part
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
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
            outputs = decoder_outputs + encoder_outputs
        else:
            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )
        # [end] model part

        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            if self.latent_num == 1:
                masked_lm_loss = self.loss_fct(lm_logits,labels,selected_logits,select_mask,latent_logits)
            else:
                masked_lm_loss = self.loss_fct(lm_logits[:,2:,:],labels[:,2:],selected_logits,select_mask,latent_logits)

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
    
    def _prepare_decoder_input_ids_for_generation(
        self, input_ids: torch.LongTensor, decoder_start_token_id: int = None, bos_token_id: int = None
    ) -> torch.LongTensor:
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        decoder_input_ids = (
            torch.ones((len(input_ids)*self.sample_num*self.latent_num, 1), dtype=torch.long, device=input_ids[0].device) * decoder_start_token_id
        )
        return decoder_input_ids

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, input_ids: torch.LongTensor,
        model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            sentence_num = [x.shape[0] for x in input_ids]
            input_ids = torch.cat(input_ids, 0)
            if self.sample_pos == 0:
                input_ids, _, attention_mask, selected_logits,select_mask, selected_idxs, selected_probs = self.selector_select(input_ids,None,model_kwargs["attention_mask"],sentence_num,"top")
                model_kwargs["attention_mask"] = attention_mask
                self.selected_idxs = selected_idxs
                self.selected_probs = selected_probs
                
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value for argument, value in model_kwargs.items() if not argument.startswith("decoder_")
            }
            encoder_outputs = encoder(input_ids, return_dict=True, **encoder_kwargs)
            if self.sample_pos == 1:
                input_ids, encoder_outputs, attention_mask, selected_logits,select_mask, selected_idxs, selected_probs = self.selector_select(input_ids,encoder_outputs,model_kwargs["attention_mask"],sentence_num,"top")
                model_kwargs["attention_mask"] = attention_mask
                self.selected_idxs = selected_idxs
                self.selected_probs = selected_probs

            model_kwargs["encoder_outputs"]: ModelOutput = encoder_outputs
        return model_kwargs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            if past[0][0].size()[2] == 2 and self.latent_num > 1:
                # decoder_input_ids = self.latent_signal.to(decoder_input_ids.device).tile(decoder_input_ids.size()[0]//self.latent_num).unsqueeze(-1)
                decoder_input_ids = self.latent_signal.to(decoder_input_ids.device).unsqueeze(-1).tile([encoder_outputs[0].size()[0], decoder_input_ids.size()[0]//self.latent_num // encoder_outputs[0].size()[0]]).reshape([-1,1])
            else:
                decoder_input_ids = decoder_input_ids[:, -1:]
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    
    #@staticmethod
    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        input_ids = input_ids.tile([expand_size,1])
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state
            model_kwargs["encoder_outputs"] = encoder_outputs
            
        return input_ids, model_kwargs


