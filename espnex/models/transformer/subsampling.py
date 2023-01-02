from typing import Callable, Optional, Sequence, Tuple, TypeVar, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import Conv, Dense, Dropout, LayerNorm, Module
from flax.linen.module import merge_param
from jax import Array
from jax.random import bernoulli

from espnet.nets.pytorch_backend.transformer.subsampling import (
    TooShortUttError,
    check_short_utt,
)
from espnex.typing import OptionalArray

from ..transformer.embedding import AddPositionalEncoding


def get_output_lengths(
    lengths,
    kernel_sizes,
    strides=1,
    paddings=0,
    dilations=1,
):
    return (lengths + 2 * paddings - dilations * (kernel_sizes - 1) - 1) // strides + 1


class Conv2dSubsampling(nn.Module):
    odim: int
    dropout_rate: float
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    pos_enc: Optional[Callable[[Array], Array]] = None
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(
        self, x: Array, ilens: OptionalArray, deterministic: Optional[bool] = None
    ) -> Tuple[Array, OptionalArray]:
        """Subsample x.

        Args:
            x (jax.Array): Input tensor (#batch, time, idim).
            ilens (jax.Array): Input lengths (#batch,).

        Returns:
            jax.Array: Subsampled tensor (#batch, time', odim),
            jax.Array: Subsampled lengths (#batch,),

        """
        deterministic = merge_param("deterministic", self.deterministic, deterministic)

        x = jnp.expand_dims(x, -1)  # (b, t, f, c=1)
        for kernel, stride in zip(self.kernel_sizes, self.strides):
            x = Conv(self.odim, [kernel] * 2, stride, padding=0)(x)  # (b, t, f, d)
            x = nn.relu(x)
        b, t, *_ = x.shape
        x = x.reshape((b, t, -1))  # (b, t, f * d)
        x = Dense(self.odim)(x)
        if self.pos_enc is None:
            x = AddPositionalEncoding(self.dropout_rate, init_type="espnet")(
                x, deterministic=deterministic
            )
        else:
            x = self.pos_enc(x)

        if ilens is not None:
            for kernel, stride in zip(self.kernel_sizes, self.strides):
                ilens = get_output_lengths(ilens, kernel, stride)
        return x, ilens


defaults_settings = {
    2: [[3, 3], [2, 1]],
    4: [[3, 3], [2, 2]],
    6: [[3, 5], [2, 3]],
    8: [[3, 3, 3], [2, 2, 2]],
}


def get_default_conv2d_subsampling(
    odim: int,
    subsample_ratio: int,
    dropout_rate: float,
    pos_enc: Optional[Module] = None,
    deterministic: Optional[bool] = None,
) -> Conv2dSubsampling:
    assert (
        subsample_ratio in defaults_settings
    ), f"subsample_ratio must be one of {list(defaults_settings.keys())}!"
    kernels, strides = defaults_settings[subsample_ratio]
    return Conv2dSubsampling(
        odim, dropout_rate, kernels, strides, pos_enc, deterministic
    )
