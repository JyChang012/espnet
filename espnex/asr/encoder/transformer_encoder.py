from typing import List, Literal, Optional, Tuple, Type, overload

import flax.linen as nn
from flax.linen import (
    Dense,
    Dropout,
    Embed,
    LayerNorm,
    Module,
    MultiHeadDotProductAttention,
    Sequential,
    make_attention_mask,
    merge_param,
    relu,
)
from jax import Array

from espnex.asr.encoder.abc import AbsEncoder
from espnex.models.transformer.embedding import AddPositionalEncoding
from espnex.models.transformer.encoder_layer import EncoderLayer
from espnex.models.transformer.multi_layer_conv import Conv1dLinear, MultiLayerConv1d
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.subsampling import get_default_conv2d_subsampling
from espnex.models.utils import make_pad_mask
from espnex.typing import OptionalArray


# from typeguard import check_argument_types

# from torch.nn.functional import ctc_loss


class TransformerEncoder(AbsEncoder):
    attention_features: int = 256
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 6
    dropout_rate: float = 0.1
    positional_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.0
    input_layer: Literal[
        "linear", "conv2d", "conv2d2", "conv2d6", "conv2d8", "embed"
    ] = "conv2d"
    num_embeddings: Optional[int] = None
    pos_enc_type: Type[Module] = AddPositionalEncoding
    normalize_before: bool = True
    concat_after: bool = False
    positionwise_layer_type: Literal["linear", "conv1d", "conv1d-linear"] = "linear"
    positionwise_conv_kernel_size: int = 1
    padding_idx: int = -1  # flax's Embed layer does not have this argument
    interctc_layer_idx: Optional[List[int]] = None  # currently ignore interctc
    interctc_use_conditioning: bool = False
    deterministic: Optional[bool] = None

    def output_size(self) -> int:
        return self.attention_features

    @overload  # for type checker
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            prev_states: Array,
            deterministic: Optional[bool]) -> Tuple[Array, Array, Array]:
        ...

    @overload
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            *,
            deterministic: Optional[bool]
    ) -> Tuple[Array, Array, None]:
        ...

    @nn.compact
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            prev_states: Optional[Array] = None,
            deterministic: Optional[bool] = None,
            # ctc: Optional[CTC] = None,
    ) -> Tuple[Array, Array, Optional[Array]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        deterministic = merge_param("deterministic", self.deterministic, deterministic)
        idim = xs_pad.shape[-1]

        # Apply embed layers
        pos_enc_layer = self.pos_enc_type(
            self.positional_dropout_rate,
            init_type="espnet",
            deterministic=deterministic,
        )
        if self.input_layer == "linear":
            xs_pad = Sequential(
                [
                    Dense(self.attention_features),
                    LayerNorm(),
                    Dropout(self.dropout_rate, deterministic=deterministic),
                    relu,
                    pos_enc_layer,
                ]
            )(xs_pad)
        elif "conv2d" in self.input_layer:
            # short utt is not checked here since it is incompatible to JIT compilation (depends on value of ilens)
            if len(self.input_layer) == len("conv2d"):
                ratio = 4
            else:
                ratio = int(self.input_layer[-1])
            xs_pad, ilens = get_default_conv2d_subsampling(
                self.attention_features, ratio, self.dropout_rate
            )(xs_pad, ilens, deterministic=deterministic)
        elif self.input_layer == "embed":
            assert isinstance(
                self.num_embeddings, int
            ), "Invalid num_embeddings or num_embeddings not given as argument."
            xs_pad = Sequential(
                [
                    Embed(self.num_embeddings, self.attention_features, dtype=float),
                    pos_enc_layer,
                ]
            )(xs_pad)
        elif self.input_layer is None and idim != self.attention_features:
            xs_pad = Dense(self.attention_features)(xs_pad)

        # apply encoder layers
        if self.positionwise_layer_type == "linear":
            positionwise_layer: Type = PositionwiseFeedForward
            positionwise_layer_args: Tuple = (
                self.linear_units,
                self.dropout_rate,
                nn.relu,
                deterministic,
            )
        elif self.positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayerConv1d
            positionwise_layer_args = (
                self.linear_units,
                self.positionwise_conv_kernel_size,
                self.dropout_rate,
                deterministic,
            )
        elif self.positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                self.linear_units,
                self.positionwise_conv_kernel_size,
                self.dropout_rate,
                deterministic,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        masks = ~make_pad_mask(ilens, xs_pad.shape[1])  # (bs, t)
        attn_masks = make_attention_mask(masks, masks, dtype=bool)  # (bs, 1, t, t)
        for layer_idx in range(self.num_blocks):
            encoder_layer = EncoderLayer(
                MultiHeadDotProductAttention(
                    self.attention_heads, dropout_rate=self.attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                self.dropout_rate,
                self.normalize_before,
                self.concat_after,
            )
            xs_pad, _ = encoder_layer(xs_pad, attn_masks, None, deterministic)
            # TODO: currently ignore interctc

        if self.normalize_before:
            xs_pad = LayerNorm()(xs_pad)

        return xs_pad, ilens, None
