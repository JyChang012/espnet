from functools import partial

import numpy as np

from espnex.models.transformer.stochastic_sequential import StochasticSequential
from flax.linen import Dense
from pytest import mark
from test.espnex.utils import *


def test_stochastic_sequential():
    layers = tuple(Dense(k) for k in [24] * 8)
    mdl = StochasticSequential(layers, .5)

    x = np.ones([3, 24])
    rngs = dict(zip(["dropout", "skip_layer", "params"], random.split(init_key, 3)))
    variables = jax.jit(partial(mdl.init, deterministic=True))(rngs, x)
    y1 = jax.jit(partial(mdl.apply, deterministic=True))(variables, x)
    y2 = jax.jit(partial(mdl.apply, deterministic=False))(variables, x, rngs=rngs)
    assert jnp.sum(y1 - y2) ** 2 > 1e-2

