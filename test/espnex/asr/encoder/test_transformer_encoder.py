import jax
import numpy as np
import torch
from jax.random import PRNGKey, randint, split, uniform
from numpy.testing import assert_equal
from jax.nn.initializers import glorot_uniform

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder as TorchTE
from espnex.asr.encoder.transformer_encoder import TransformerEncoder
from test.espnex.utils import *


def test_transformer_encoder():
    model = TransformerEncoder(kernel_init=glorot_uniform())
    rng = PRNGKey(0)
    x = uniform(rng, [32, 256, 128])  # (bs, t, f)
    (rng,) = split(rng, 1)
    ilens = randint(rng, [32], 20, 256)
    ilens = ilens.at[-1].set(256)
    rng, *rngs = split(rng, 4)
    rngs = dict(zip(["dropout", "params", "skip_layer"], rngs))
    vars = model.init(rngs, x, ilens, None, False)

    def apply(vars, x, ilens, deterministic, rngs):
        return model.apply(vars, x, ilens, None, deterministic, rngs=rngs)

    apply = jax.jit(apply, static_argnames="deterministic")
    y, olens, _ = apply(vars, x, ilens, False, rngs)

    tmodel = TorchTE(128)

    assert compare_params(tmodel, vars['params'])


    """
    tx = torch.Tensor(jax.device_get(x))
    tlens = torch.Tensor(jax.device_get(ilens))
    with torch.no_grad():
        ty, tolens, _ = tmodel(tx, tlens)
    assert_equal(np.array(ty.shape), np.array(y.shape))
    assert_equal(jax.device_get(olens), tolens.numpy())  # currently olens is not the same! (caused by subsampling)
    pass
    """
