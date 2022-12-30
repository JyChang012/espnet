import math
from typing import Literal, Optional, Sequence, no_type_check

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import Dense, Dropout, LayerNorm, Module
from flax.linen.module import merge_param
from jax import Array
from jax.random import bernoulli


def sinusoidal_init(
    max_len=2048, min_scale=1.0, max_scale=10000.0, init_type="default"
):
    """1D Sinusoidal Position Embedding Initializer. Copy from https://github.com/google/flax/tree/main/examples
    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.
    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(shape: Sequence[int], dtype=np.float32) -> Array:
        """Sinusoidal init."""
        del dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    def init_espnet(shape: Sequence[int], dtype=np.float32) -> Array:
        """Sinusoidal init."""
        del dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)  # (max_len, d_feature)
        position = np.arange(0, max_len)[:, np.newaxis]  # (max_len, 1)
        scale_factor = -np.log(max_scale / min_scale) / d_feature
        div_term = min_scale * np.exp(
            np.arange(0, d_feature, 2) * scale_factor
        )  # (d_feature // 2,)
        pe[:, ::2] = np.sin(position * div_term)  # (max_len, d_feature // 2)
        if d_feature % 2:
            div_term = div_term[:-1]
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    if init_type == "default":
        return init
    else:
        return init_espnet


class AddPositionalEncoding(Module):
    dropout_rate: float
    max_len: int = 5000  # initial lengths of positional encoding
    reverse: bool = False  # currently not used
    init_type: Literal["default", "espnet"] = "default"
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, deterministic: Optional[bool] = None) -> Array:
        """Add positional encoding. Dynamically increase T of positional encoding. Might have performance issue when using with JIT

        Args:
            x (jax.Array): Input tensor (batch, time, `*`).

        Returns:
            jax.Array: Encoded tensor (batch, time, `*`).
        """
        deterministic = merge_param("deterministic", deterministic, self.deterministic)
        length = x.shape[1]
        max_len = max(self.max_len, length)
        pos_emb_shape = (1, max_len, x.shape[-1])
        pos_emb = sinusoidal_init(max_len=max_len, init_type=self.init_type)(
            pos_emb_shape
        )
        x_scale = np.sqrt(x.shape[-1])
        x = x * x_scale + pos_emb[:, :length, :]
        x = Dropout(self.dropout_rate)(x, deterministic)
        return x
