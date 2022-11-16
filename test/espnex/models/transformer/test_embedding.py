import jax
import jax.numpy as jnp
import torch
from jax.random import PRNGKey
from numpy.testing import assert_allclose
from pytest import mark

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnex.models.transformer.embedding import AddPositionalEncoding

rtol = 1e-6
atol = 1e-6


@mark.parametrize("b", [1, 5])
@mark.parametrize("t", [2, 8])
@mark.parametrize("d", [2, 6])
@mark.parametrize("lens", [[10, 20]])
@mark.parametrize("deterministic", [True, False])
@mark.parametrize("jit", [True, False])
@mark.parametrize("init_type", ["espnet", "default"])
def test_embedding(b, t, d, lens, deterministic, jit, init_type):
    le, ext_le = lens
    t = min(le, t)
    x = jnp.zeros([b, t, d], dtype=float)
    model = AddPositionalEncoding(dropout_rate=0.5, max_len=le, init_type=init_type)

    rngs = {"dropout": PRNGKey(0)}

    def apply(x, rngs):
        return model.apply({}, x, deterministic, rngs=rngs)

    if jit:
        apply = jax.jit(apply)

    y = apply(x, rngs)

    # test auto extend embeddings length, supported by the torch version
    ext_x = jnp.empty([b, ext_le, d])
    ext_y = apply(ext_x, rngs)
    if deterministic and init_type == "espnet":
        torch_encoder = PositionalEncoding(d, 0.5, le)
        torch_encoder.eval()
        x = torch.zeros([b, t, d])
        ty = torch_encoder(x)
        assert_allclose(jax.device_get(y), ty.numpy(), rtol=rtol, atol=atol)
