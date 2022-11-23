import jax

from espnex.models.transformer.encoder_layer import EncoderLayer
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from flax.linen import MultiHeadDotProductAttention
import jax.numpy as jnp
from jax.random import PRNGKey, split, bernoulli
from functools import partial
from numpy.testing import assert_allclose


def test_encoder_layer():
    attn = MultiHeadDotProductAttention(num_heads=4, dropout_rate=.5)
    ffn = PositionwiseFeedForward(256, .5)
    model = EncoderLayer(attn, ffn, .5, stochastic_depth_rate=.5)
    key = PRNGKey(0)
    xrng, mask_rng, *rngs = split(key, 5)
    x = jax.random.normal(xrng, [5, 128, 256]) * 4
    mask = bernoulli(mask_rng, .5, [5, 1, 128, 128])
    rngs = dict(zip(['dropout', 'skip_layer', 'params'], rngs))
    variables = model.init(rngs, x, mask, None, False)

    def apply(vars, x, mask, rngs, deterministic):
        return model.apply(vars, x, mask, None, deterministic, rngs=rngs)

    apply = jax.jit(apply, static_argnames='deterministic')

    rngs['skip_layer'] = PRNGKey(0)
    y, ymask = apply(variables, x, mask, rngs, False)
    assert_allclose(jax.device_get(x), jax.device_get(y))

    rngs['skip_layer'] = PRNGKey(3)
    y, ymask = apply(variables, x, mask, rngs, False)
    assert jnp.linalg.norm(x - y) > 1e-2

    rngs['skip_layer'] = PRNGKey(0)
    y, ymask = apply(variables, x, mask, rngs, True)
    assert jnp.linalg.norm(x - y) > 1e-2
