import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split
from pytest import mark

from espnex.models.transformer.subsampling import (
    defaults_settings,
    get_default_conv2d_subsampling,
)

key = PRNGKey(15213)
rngs = split(key, 2)
rngs = dict(zip(["params", "dropout"], rngs))


@mark.parametrize("in_shape", [[1, 256, 128]])  # (b, t, f)
@mark.parametrize("ratio", defaults_settings)
@mark.parametrize("odim", [1, 5])
@mark.parametrize("deterministic", [True, False])
@mark.parametrize("jit", [True, False])
def test_subsampling(in_shape, ratio, odim, deterministic, jit):
    model = get_default_conv2d_subsampling(odim, ratio, 0.5)
    x = jnp.zeros(in_shape)
    lengths = jax.random.randint(key, (in_shape[0],), 1, in_shape[1])
    variables = model.init(rngs, x, lengths, deterministic)

    def apply(v, x, lengths, rngs):
        return model.apply(v, x, lengths, deterministic, rngs=rngs)

    if jit:
        apply = jax.jit(apply)
    y, y_lens = apply(variables, x, lengths, rngs)
    y = apply(variables, x, None, rngs)
    pass
    # TODO: add equality test
