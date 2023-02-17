from functools import partial

import jax
import jax.numpy as jnp
from flax.linen import MultiHeadDotProductAttention, make_attention_mask
from jax.random import PRNGKey, bernoulli, split
from numpy.testing import assert_allclose

from espnex.models.transformer.encoder_layer import EncoderLayer
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward


def test_encoder_layer():
    def get_model(d=False):
        attn = MultiHeadDotProductAttention(num_heads=4, dropout_rate=0.5, deterministic=d)
        ffn = PositionwiseFeedForward(256, 0.5, deterministic=d)
        model = EncoderLayer(attn, ffn, 0.5, stochastic_depth_rate=0.5)
        return model
    model = get_model()
    key = PRNGKey(0)
    xrng, mask_rng, *rngs = split(key, 5)
    x = jax.random.normal(xrng, [5, 128, 256]) * 4
    mask = bernoulli(mask_rng, 0.5, [5, 128])
    mask = make_attention_mask(mask, mask, dtype=bool)

    rngs = dict(zip(["dropout", "skip_layer", "params"], rngs))
    variables = model.init(rngs, x, mask, None, False)

    def apply(vars, x, mask, rngs, deterministic):
        return get_model(deterministic).apply(vars, x, mask, None, deterministic, rngs=rngs)

    apply = jax.jit(apply, static_argnames="deterministic")

    rngs["skip_layer"] = PRNGKey(0)
    y, ymask = apply(variables, x, mask, rngs, False)
    assert_allclose(jax.device_get(x), jax.device_get(y))

    rngs["skip_layer"] = PRNGKey(3)
    y, ymask = apply(variables, x, mask, rngs, False)
    assert jnp.linalg.norm(x - y) > 1e-2

    rngs["skip_layer"] = PRNGKey(0)
    y, ymask = apply(variables, x, mask, rngs, True)
    assert jnp.linalg.norm(x - y) > 1e-2
