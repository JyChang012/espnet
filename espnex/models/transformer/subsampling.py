from typing import Optional, Sequence, Union, Tuple

import flax.linen as nn
from flax.linen import LayerNorm, Dense, Dropout, Module, Conv
import jax
import jax.numpy as jnp
from flax.linen.module import merge_param
from jax import Array
from jax.random import bernoulli

from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt, TooShortUttError
from ..transformer.embedding import AddPositionalEncoding


class Conv2dSubsampling(nn.Module):
    odim: int
    dropout_rate: float
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    pos_enc: Optional[Module] = None

    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array], deterministic: bool) -> Union[Tuple[Array, Array], Array]:
        """Subsample x.

        Args:
            x (jax.Array): Input tensor (#batch, time, idim).
            mask (jax.Array): Input mask (#batch, time).

        Returns:
            jax.Array: Subsampled tensor (#batch, time', odim),
            jax.Array: Subsampled mask (#batch, time'),

        """
        x = jnp.expand_dims(x, -1)  # (b, t, f, c=1)
        for kernel, stride in zip(self.kernel_sizes, self.strides):
            x = Conv(self.odim, [kernel], stride, padding=0)(x)  # (b, t, f, d)
            x = nn.relu(x)
        b, t, *_ = x.shape
        x = x.reshape((b, t, -1))  # (b, t, f * d)
        x = Dense(self.odim)(x)
        if self.pos_enc is None:
            x = AddPositionalEncoding(self.dropout_rate, init_type='espnet')(x, deterministic)
        else:
            x = self.pos_enc(x)
        if mask is None:
            return x
        else:
            for kernel, stride in zip(self.kernel_sizes, self.strides):
                mask = mask[:, :-(kernel - 1):stride]
            return x, mask


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
        pos_enc: Optional[Module] = None
) -> Conv2dSubsampling:
    assert subsample_ratio in defaults_settings, f'subsample_ratio must be one of {list(defaults_settings.keys())}!'
    kernels, strides = defaults_settings[subsample_ratio]
    return Conv2dSubsampling(odim, dropout_rate, kernels, strides, pos_enc)
