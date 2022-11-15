from typing import Callable

import flax.linen as nn
from flax.linen import Module, Dense, Dropout
from jax import Array


class PositionwiseFeedForward(Module):
    """Position-wise feed forward layer.

    Args:
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """
    hidden_units: int
    dropout_rate: float
    activation: Callable[[Array], Array] = nn.relu

    @nn.compact
    def __call__(self, x: Array, deterministic: bool = False) -> Array:
        in_dim = x.shape[-1]
        x = Dense(self.hidden_units)(x)
        x = self.activation(x)
        x = Dropout(self.dropout_rate)(x, deterministic)
        x = Dense(in_dim)(x)
        return x

