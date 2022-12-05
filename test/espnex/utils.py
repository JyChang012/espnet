from typing import Union

import numpy
from jax.random import PRNGKey, split, fold_in, normal, uniform, randint, bernoulli
import jax.random as random
import jax
import torch
from numpy.testing import assert_equal, assert_allclose


init_key = PRNGKey(15213)


def j2t(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to torch Tensor."""
    return torch.tensor(jax.device_get(arr))


def a2n(arr: Union[jax.Array, torch.Tensor]) -> numpy.ndarray:
    """Convert array to numpy array"""
    if isinstance(arr, jax.Array):
        return jax.device_get(arr)
    elif isinstance(arr, torch.Tensor):
        return arr.numpy()
    else:
        return arr



