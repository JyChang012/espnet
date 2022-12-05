from jax.random import PRNGKey, split, fold_in, normal, uniform, randint, bernoulli
import jax.random as random
import jax
import torch
from numpy.testing import assert_equal, assert_allclose


init_key = PRNGKey(15213)


def j2t(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to torch Tensor."""
    return torch.tensor(jax.device_get(arr))

