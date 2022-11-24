import math
from typing import Literal, Optional

import numpy as np
import flax.linen as nn
from flax.linen import LayerNorm, Dense, Dropout, Module
import jax
import jax.numpy as jnp
from flax.linen.module import merge_param
from jax.random import bernoulli
from jax import Array


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0,
                    init_type='default'):
    """1D Sinusoidal Position Embedding Initializer. Copy from https://github.com/google/flax/tree/main/examples
    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.
    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, :d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    def init_espnet(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / d_feature
        div_term = min_scale * np.exp(np.arange(0, d_feature, 2) * scale_factor)
        pe[:, ::2] = np.sin(position * div_term)
        if div_term.shape[0] % 2:
            div_term = div_term[:-1]
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    if init_type == 'default':
        return init
    else:
        return init_espnet


class AddPositionalEncoding(Module):
    dropout_rate: float
    max_len: int = 5000  # initial lengths of positional encoding
    reverse: bool = False  # currently not used
    init_type: Literal['default', 'espnet'] = 'default'
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, deterministic: Optional[bool] = None) -> Array:
        """Add positional encoding. Dynamically increase T of positional encoding. Might have performance issue when using with JIT

        Args:
            x (jax.Array): Input tensor (batch, time, `*`).

        Returns:
            jax.Array: Encoded tensor (batch, time, `*`).
        """
        deterministic = merge_param('deterministic', deterministic, self.deterministic)
        length = x.shape[1]
        max_len = max(self.max_len, length)
        pos_emb_shape = (1, max_len, x.shape[-1])
        pos_emb = sinusoidal_init(max_len=max_len, init_type=self.init_type)(None, pos_emb_shape, None)
        x_scale = np.sqrt(x.shape[-1])
        x = x * x_scale + pos_emb[:, :length, :]
        x = Dropout(self.dropout_rate)(x, deterministic)
        return x


