from typing import Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact
from flax.linen.module import compact, merge_param
from jax import Array
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.multi_layer_conv import MultiLayerConv1d, Conv1dLinear


class DecoderLayer(Module):
    self_attn: MultiHeadDotProductAttention
    src_attn: MultiHeadDotProductAttention
    feed_forward: Union[PositionwiseFeedForward, MultiLayerConv1d, Conv1dLinear]
    dropout_rate: float
    normalize_before: bool = True
    concat_after: bool = False
    deterministic: Optional[bool] = None

    @compact
    def __call__(
            self,
            inputs: Array,
            decoder_mask: Array,
            encoded: Array,
            encoder_decoder_mask: Array,
            deterministic: Optional[bool] = None
    ):
        deterministic = merge_param('deterministic', deterministic, self.deterministic)

        x = inputs
        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)

        feat_size = inputs.shape[-1]

        def attn(inp_q, inp_kv, mask, attn_func):
            out = attn_func(inp_q, inp_kv, mask)
            if self.concat_after:
                out = jnp.concatenate((x, out), axis=-1)
                out = Dense(feat_size)(out)
            else:
                out = Dropout(self.dropout_rate, deterministic=deterministic)(out)
            return out

        # casual self-attention
        x = attn(x, x, decoder_mask, self.self_attn)
        x = x + residual
        if not self.normalize_before:
            x = LayerNorm()(x)

        # cross attention
        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)
        x = attn(x, encoded, encoder_decoder_mask, self.src_attn)
        x = x + residual
        if not self.normalize_before:
            x = LayerNorm()(x)

        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)
        x = residual + Dropout(self.dropout_rate, deterministic=deterministic)(self.feed_forward(x))
        if not self.normalize_before:
            x = LayerNorm()(x)

        return x




