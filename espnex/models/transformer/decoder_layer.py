from typing import Callable, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact
from flax.linen.module import compact, merge_param
from jax import Array
from jax.nn.initializers import Initializer

from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.multi_layer_conv import MultiLayerConv1d, Conv1dLinear
from espnex.models.utils import inject_args


class DecoderLayer(Module):
    self_attn: MultiHeadDotProductAttention
    src_attn: MultiHeadDotProductAttention
    feed_forward: Union[PositionwiseFeedForward, MultiLayerConv1d, Conv1dLinear]
    dropout_rate: float
    normalize_before: bool = True
    concat_after: bool = False
    kernel_init: Optional[Initializer] = None

    @compact
    def __call__(
            self,
            inputs: Array,
            encoded: Array,
            deterministic: bool,
            self_attention_mask: Optional[Array] = None,
            cross_attention_mask: Optional[Array] = None,
    ) -> Array:
        """
        Args:
            inputs (Array): Inputs to decoder. Float array of shape (bs, dec_maxlen, dec_feat_len), or (bs, 1, feat_len) when in single step
                `decode` mode
            encoded (Array): Encoded output from encoder. Float array of shape (bs, enc_maxlen, enc_feat_len)
            self_attention_mask (Array): self attention mask of shape (bs, n_heads, dec_maxlen, dec_maxlen), normally
                a causal mask when in train mode. Could be `None` when in single step `decode` mode, in which case a
                causal mask will be auto generated based on `cached_index`.
            cross_attention_mask (Array): cross attention mask of shape (bs, n_heads, dec_maxlen, enc_maxlen)
            deterministic (bool): whether to disable dropout

        Returns:
            output of shape (batch_size, dec_maxlen, out_feat) or (batch_size, 1, out_feat) when in single step
            `decode` mode

        """
        x = inputs
        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)

        feat_size = inputs.shape[-1]

        def attn(inp_q, inp_kv, mask, attn_func):
            out = attn_func(inp_q, inp_kv, mask, deterministic=deterministic)
            if self.concat_after:
                out = jnp.concatenate((x, out), axis=-1)
                dense = inject_args(Dense, kernel_init=self.kernel_init)
                out = dense(feat_size)(out)
            else:
                out = Dropout(self.dropout_rate, deterministic=deterministic)(out)
            return out

        # casual self-attention
        x = attn(x, x, self_attention_mask, self.self_attn)
        x = x + residual
        if not self.normalize_before:
            x = LayerNorm()(x)

        # cross attention
        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)
        x = attn(x, encoded, cross_attention_mask, self.src_attn)
        x = x + residual
        if not self.normalize_before:
            x = LayerNorm()(x)

        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)
        x = residual + Dropout(self.dropout_rate,
                               deterministic=deterministic)(self.feed_forward(x,
                                                                              deterministic=deterministic))
        if not self.normalize_before:
            x = LayerNorm()(x)

        return x




