from typing import Union, Dict, Any, Optional, Mapping

import numpy as np
from jax.random import PRNGKey, split, fold_in, normal, uniform, randint, bernoulli
import jax.random as random
import jax
import jax.numpy as jnp
from jax import tree_util
import torch
from numpy.testing import assert_equal, assert_allclose


init_key = PRNGKey(15213)


def j2t(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to torch Tensor."""
    return torch.tensor(jax.device_get(arr))


def a2n(arr: Union[jax.Array, torch.Tensor]) -> np.ndarray:
    """Convert array to numpy array"""
    if isinstance(arr, jax.Array):
        return jax.device_get(arr)
    elif isinstance(arr, torch.Tensor):
        return arr.numpy()
    else:
        return arr


def count_params(mdl: Union[torch.nn.Module, Mapping[str, Any]]) -> Optional[int]:
    ret = None
    if isinstance(mdl, torch.nn.Module):
        ret = 0
        for p in mdl.parameters():
            ret += p.numel()
    elif isinstance(mdl, Mapping):
        mdl = tree_util.tree_leaves(mdl)
        mdl = tree_util.tree_map(lambda x: x.size, mdl)
        ret = sum(mdl)
    return ret


def compare_params(mdl1, mdl2):
    return count_params(mdl1) == count_params(mdl2)






