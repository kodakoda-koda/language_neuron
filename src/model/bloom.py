from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BloomConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.bloom.modeling_bloom import (
    BloomAttention,
    BloomBlock,
    BloomForCausalLM,
    BloomMLP,
    BloomModel,
    dropout_add,
)

from src.model.outputs import CustomBaseModelOutputWithPastAndCrossAttentions, CustomCausalLMOutputWithCrossAttentions


class CustomBloomAttention(BloomAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_neurons: bool = False,
    ):
        batch_size, q_length, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        attn_neurons = torch.tensor([]).to(device=device, dtype=dtype) if output_neurons else None

        fused_qkv = self.query_key_value(hidden_states)
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)

        # for detecting neurons
        if output_neurons:
            query_layer_ = query_layer.reshape(batch_size, q_length, -1)
            key_layer_ = key_layer.reshape(batch_size, q_length, -1)
            value_layer_ = value_layer.reshape(batch_size, q_length, -1)
            attn_neurons = torch.cat([attn_neurons, query_layer_, key_layer_, value_layer_], dim=-1)

        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim).transpose(-1, -2)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask

        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)

        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, -1)

        context_layer = torch.bmm(attention_probs_reshaped, value_layer)

        context_layer = self._merge_heads(context_layer)

        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = self.hidden_size / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + F.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        # for detecting neurons
        if output_neurons:
            output_tensor_ = output_tensor.view(batch_size, q_length, -1)
            attn_neurons = torch.cat([attn_neurons, output_tensor_], dim=-1)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)
        if output_neurons:
            outputs += (attn_neurons,)

        return outputs


class CustomBloomMLP(BloomMLP):
    def forward(
        self, hidden_states: torch.Tensor, residual: torch.Tensor, output_neurons: bool = False
    ) -> torch.Tensor:
        device = hidden_states.device
        dtype = hidden_states.dtype
        mlp_neurons = torch.tensor([]).to(device=device, dtype=dtype) if output_neurons else None

        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        # for detecting neurons
        if output_neurons:
            mlp_neurons = torch.cat([mlp_neurons, hidden_states], dim=-1)

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + F.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        # for detecting neurons
        if output_neurons:
            mlp_neurons = torch.cat([mlp_neurons, intermediate_output], dim=-1)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)
        outputs = (output, mlp_neurons)

        return outputs


class CustomBloomBlock(BloomBlock):
    def __init__(self, config: BloomConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.self_attention = CustomBloomAttention(config, layer_idx)
        self.mlp = CustomBloomMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_past: Optional[Cache] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        output_neurons: bool = False,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]
        device = hidden_states.device
        dtype = hidden_states.dtype
        all_neurons = torch.tensor([]).to(device=device, dtype=dtype) if output_neurons else None

        layernorm_output = self.input_layernorm(hidden_states)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
            output_neurons=output_neurons,
        )

        attention_output = attn_outputs[0]
        layernorm_output = self.post_attention_layernorm(attention_output)

        # for detecting neurons
        if output_neurons:
            all_neurons = torch.cat([all_neurons, attn_outputs[-1]], dim=-1)
            outputs = attn_outputs[1:-1]
        else:
            outputs = attn_outputs[1:]

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        mlp_outputs = self.mlp(layernorm_output, residual, output_neurons)

        if use_cache:
            outputs = (mlp_outputs[0],) + outputs
        else:
            outputs = (mlp_outputs[0],) + outputs[1:]

        # for detecting neurons
        if output_neurons:
            all_neurons = torch.cat([all_neurons, mlp_outputs[-1]], dim=-1)

        outputs = outputs + (all_neurons,)

        return outputs  # hidden_states, past_kv, attentions, neurons


class CustomBloomModel(BloomModel):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.h = nn.ModuleList([CustomBloomBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_neurons: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor, ...], CustomBaseModelOutputWithPastAndCrossAttentions]:
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        batch_size, seq_length, _ = inputs_embeds.shape
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        seq_length_with_past = seq_length + past_length
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + seq_length, device=inputs_embeds.device)

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # for detecting neurons
        device = hidden_states.device
        dtype = hidden_states.dtype
        all_neurons = torch.tensor([]).to(device=device, dtype=dtype) if output_neurons else None

        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    alibi,
                    causal_mask,
                    past_key_values,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    cache_position,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=past_key_values,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                    cache_position=cache_position,
                    output_neurons=output_neurons,
                )

            hidden_states = outputs[0]
            if use_cache:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # for detecting neurons
            if output_neurons:
                all_neurons = torch.cat([all_neurons, outputs[-1]], dim=-1)

        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # for detecting neurons
        if output_neurons:
            pad_token_indices = torch.where(input_ids == self.config.pad_token_id)
            all_neurons[pad_token_indices[0], pad_token_indices[1], :] = torch.nan
            all_neurons = torch.nanmean(all_neurons, dim=1)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )

        return CustomBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            neurons=all_neurons,
        )


class CustomBloomForCausalLM(BloomForCausalLM):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.transformer = CustomBloomModel(config)

        self.post_init()

    # ToDo: add intervention
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_neurons: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[Tuple[torch.Tensor], CustomCausalLMOutputWithCrossAttentions]:
        num_items_in_batch = deprecated_arguments.pop("num_items_in_batch", None)
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            output_neurons=output_neurons,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                num_items_in_batch=num_items_in_batch,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CustomCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            neurons=transformer_outputs.neurons,
        )
