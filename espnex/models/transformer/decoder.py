from typing import Callable, Optional, Tuple, Union, Type, Literal, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact, Sequential, Embed
from flax.linen.module import compact, merge_param
from flax.linen.activation import relu
from jax import Array

from espnex.models.transformer.decoder_layer import DecoderLayer
from espnex.models.transformer.embedding import AddPositionalEncoding
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.multi_layer_conv import MultiLayerConv1d, Conv1dLinear


class Decoder(Module):
    odim: int  # vocabulary size
    selfattn_layer_type: Literal['selfattn'] = 'selfattn'
    # TODO: Add lightconv and dynamicconv
    attention_dim: int = 256
    attention_heads: int = 4
    conv_wshare: int = 4
    conv_kernel_length: int = 11
    conv_usebias: bool = False
    linear_units: int = 2048
    num_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    self_attention_dropout_rate: float = 0.0
    src_attention_dropout_rate: float = 0.0
    input_layer: Union[Module, Callable[..., Any], Literal['embed', 'linear']] = "embed"
    use_output_layer: bool = True
    pos_enc_class: Type[AddPositionalEncoding] = AddPositionalEncoding
    normalize_before: bool = True
    concat_after: bool = False
    deterministic: Optional[bool] = None

    @compact
    def __call__(
            self,
            inputs: Array,
            encoded: Array,
            self_attention_mask: Optional[Array] = None,
            cross_attention_mask: Optional[Array] = None,
            decode: bool = False,
            deterministic: Optional[bool] = None
    ) -> Array:
        """
        Args:
        Args:
            inputs (Array): Inputs to decoder. Float array of shape (bs, dec_maxlen, dec_feat_len), or (bs, 1, feat_len) when in single step
                `decode` mode
            encoded (Array): Encoded output from encoder. Float array of shape (bs, enc_maxlen, enc_feat_len)
            self_attention_mask (Array): self attention mask of shape (bs, n_heads, dec_maxlen, dec_maxlen), normally
                a causal mask when in train mode. Could be `None` when in single step `decode` mode, in which case a
                causal mask will be auto generated based on `cached_index`.
            cross_attention_mask (Array): cross attention mask of shape (bs, n_heads, dec_maxlen, enc_maxlen)
            decode: whether in `decode` mode
            deterministic (bool): whether to disable dropout

        Returns:
            output of shape (batch_size, dec_maxlen, out_feat) or (batch_size, 1, out_feat) when in single step
            `decode` mode
        """
        deterministic = merge_param('deterministic', self.deterministic, deterministic)
        # 1. construct embed layer
        if self.input_layer == 'embed':
            embed = Sequential((
                Embed(self.odim, self.attention_dim),
                self.pos_enc_class(self.positional_dropout_rate, deterministic=deterministic)
            ))
        elif self.input_layer == "linear":
            embed = Sequential((
                Dense(self.attention_dim),
                LayerNorm(),
                Dropout(self.dropout_rate, deterministic=deterministic),
                relu,
                self.pos_enc_class(self.positional_dropout_rate, deterministic=deterministic)
            ))
        elif isinstance(self.input_layer, (Module, Callable)):
            embed = Sequential((
                self.input_layer,
                self.pos_enc_class(self.positional_dropout_rate, deterministic=deterministic)
            ))
        else:
            raise NotImplementedError("only `embed` or `linear` or flax Module is supported.")

        # 2. construct causal self-attention layer
        if self.selfattn_layer_type == "selfattn":
            decoder_selfattn_layer = MultiHeadDotProductAttention
            decoder_selfattn_layer_args = dict(
                num_heads=self.attention_heads, 
                dropout_rate=self.self_attention_dropout_rate,
                deterministic=deterministic,
                decode=decode
            )
        else:
            raise NotImplementedError("Only support `selfattn` as `selfattention_layer_type`.")

        # 3. construct cross attention and decoder layers
        decoders = tuple(
            DecoderLayer(
                decoder_selfattn_layer(**decoder_selfattn_layer_args),
                MultiHeadDotProductAttention(
                    self.attention_heads,
                    dropout_rate=self.src_attention_dropout_rate,
                    deterministic=deterministic
                ),
                PositionwiseFeedForward(self.linear_units, self.dropout_rate, deterministic=deterministic),
                self.dropout_rate,
                self.normalize_before,
                self.concat_after,
                deterministic
            ) for _ in range(self.num_blocks)
        )

        # 4. apply the layers
        x = embed(inputs)
        for decoder in decoders:
            x = decoder(x, encoded, self_attention_mask, cross_attention_mask)
        if self.normalize_before:
            x = LayerNorm()(x)
        if self.use_output_layer:
            x = Dense(self.odim)(x)
        return x






