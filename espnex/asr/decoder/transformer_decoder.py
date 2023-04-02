import logging
from functools import partial
from typing import Callable, Optional, Tuple, Union, Type, Literal, Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact, Sequential, Embed
from flax.linen.module import compact, merge_param
from flax.linen.activation import relu
from jax import Array
from jax.nn.initializers import Initializer, glorot_uniform
from flax.linen.attention import make_causal_mask, combine_masks, make_attention_mask

from espnex.models.utils import inject_args, make_pad_mask
from espnex.models.transformer.decoder_layer import DecoderLayer
from espnex.models.transformer.embedding import AddPositionalEncoding
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.stochastic_sequential import StochasticSequential
from espnex.asr.decoder.abc import AbsDecoder


class TransformerDecoder(AbsDecoder):
    vocab_size: int
    encoder_output_size: int
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    self_attention_dropout_rate: float = 0.0
    src_attention_dropout_rate: float = 0.0
    input_layer: Literal['embed', 'linear'] = "embed"
    use_output_layer: bool = True
    pos_enc_class: Module = AddPositionalEncoding
    normalize_before: bool = True
    concat_after: bool = False
    layer_drop_rate: float = 0.0
    weight_tying: bool = False
    kernel_init: Optional[Initializer] = None

    def setup(self) -> None:
        dense = inject_args(Dense, kernel_init=self.kernel_init)
        mha = inject_args(MultiHeadDotProductAttention,
                          num_heads=self.attention_heads,
                          out_features=self.encoder_output_size,
                          qkv_features=self.encoder_output_size,
                          kernel_init=self.kernel_init)
        ffn = inject_args(PositionwiseFeedForward,
                          kernel_init=self.kernel_init)
        if self.input_layer == 'embed':
            self.embed = StochasticSequential((
                Embed(self.vocab_size, self.encoder_output_size),
                self.pos_enc_class(self.positional_dropout_rate)
            ))
        elif self.input_layer == 'linear':
            self.embed = StochasticSequential((
                dense(self.encoder_output_size),
                LayerNorm(),
                Dropout(self.dropout_rate),
                relu,
                self.pos_enc_class(self.positional_dropout_rate),
            ))
        else:
            raise ValueError(f"only 'embed' or 'linear' is supported: {self.input_layer}")

        if self.normalize_before:
            self.after_norm = LayerNorm()
        if self.use_output_layer:
            if self.weight_tying:
                assert self.input_layer == 'embed', "To use weight typing, input_layer must be `embed`!"
                if self.is_initializing():
                    logging.info(f'TransformerDecoder: {self.name} is using weight tying.')
                self.output_layer = self.embed.layers[0].attend
            else:
                self.output_layer = dense(self.vocab_size)

        self.decoders = StochasticSequential(
            tuple(
                DecoderLayer(
                    mha(dropout_rate=self.self_attention_dropout_rate),
                    mha(dropout_rate=self.src_attention_dropout_rate),
                    ffn(self.linear_units, self.dropout_rate),
                    dropout_rate=self.dropout_rate,
                    normalize_before=self.normalize_before,
                    concat_after=self.concat_after,
                    kernel_init=self.kernel_init
                ) for _ in range(self.num_blocks)
            ),
            self.layer_drop_rate
        )

    def __call__(self,
                 inputs: Array,
                 input_lengths: Array,
                 encoded: Array,
                 encoded_lengths: Array,
                 deterministic: bool,
                 decode: bool = False) -> Tuple[Array, Array]:
        """
        Args:
            inputs:  (b, ilen)
            input_lengths: (b,)
            encoded: (b, elen, efeat)
            encoded_lengths: (b)
            decode: bool
            deterministic: optional bool

        Returns: (b, ilen), (b,)
        """
        self_pad_mask = ~make_pad_mask(input_lengths, inputs.shape[1])
        encoded_pad_mask = ~make_pad_mask(encoded_lengths, encoded.shape[1])

        # make self attention mask
        self_attention_mask = make_attention_mask(self_pad_mask, self_pad_mask)
        causal_mask = make_causal_mask(inputs if len(inputs.shape) == 2 else inputs[..., 0])
        self_attention_mask = combine_masks(causal_mask, self_attention_mask)

        # make cross attention mask
        cross_attention_mask = make_attention_mask(self_pad_mask, encoded_pad_mask)

        x = self.embed(inputs, deterministic=deterministic)

        x = self.decoders(x,
                          deterministic=deterministic,
                          encoded=encoded,
                          self_attention_mask=self_attention_mask,
                          cross_attention_mask=cross_attention_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        return x, input_lengths
