from typing import Callable, Optional

import flax.linen as nn
from flax.linen import Dense, Dropout, Module, merge_param
from jax import Array
from jax.nn.initializers import Initializer

from espnex.models.utils import inject_args


class PositionwiseFeedForward(Module):
    """Position-wise feed forward layer.

    Args:
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
    """

    hidden_units: int
    dropout_rate: float
    activation: Callable[[Array], Array] = nn.relu
    kernel_init: Optional[Initializer] = None
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, deterministic: Optional[bool] = None) -> Array:
        deterministic = merge_param("deterministic", self.deterministic, deterministic)
        in_dim = x.shape[-1]

        dense = inject_args(Dense, kernel_init=self.kernel_init)

        x = dense(self.hidden_units)(x)
        x = self.activation(x)
        x = Dropout(self.dropout_rate)(x, deterministic)
        x = dense(in_dim)(x)
        return x
