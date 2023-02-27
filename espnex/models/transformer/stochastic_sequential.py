from functools import partial
from typing import Callable, Optional, Tuple, Union, Type, Literal, Any, Sequence
from inspect import signature

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
from flax.linen import Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact, Sequential, Embed
from flax.linen.module import compact, merge_param
from flax.linen.activation import relu
from flax.linen import Sequential
from jax import Array
from jax.nn.initializers import Initializer, glorot_uniform
from jax import lax
from jax import random

from espnex.models.utils import inject_args


class StochasticSequential(Module):
    layers: Union[Sequence[Callable[..., Any]], Callable[..., Any]]
    drop_rate: float = 0.
    rng_collection: str = 'skip_layer'

    def __call__(self,
                 *args: Any,
                 deterministic: bool,
                 **kwargs: Any) -> Any:
        if len(args) == 1:
            args, = args
        outputs = args
        kwargs['deterministic'] = deterministic
        deterministic = deterministic or self.drop_rate <= 0.

        is_seq = isinstance(self.layers, Sequence)
        n_layers = len(self.layers) if is_seq else 1

        dropped = deterministic or jax.random.bernoulli(self.make_rng(self.rng_collection),
                                                        self.drop_rate,
                                                        [n_layers])
        for i in range(n_layers):
            def apply_fn(mdl):  # must be a pure function. Note that callables might not be pure
                f = mdl.layers[i] if is_seq else mdl.layers
                f = inject_args(f, **kwargs)
                return f(*outputs) if isinstance(outputs, tuple) else f(outputs)

            if not deterministic:
                outputs = cond(dropped[i], lambda mdl: outputs, apply_fn, self)
            else:
                outputs = apply_fn(self)
        return outputs
