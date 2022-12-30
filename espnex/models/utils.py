from typing import Optional

import jax.numpy as jnp
from jax import Array


def make_pad_mask(  # a simplified version of the torch one, to support JIT compilation.
    lengths: Array, maxlen: int, axis: int = -1
) -> Array:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batches of lengths (*B,).
        maxlen (int): Length of the returned mask array


    Returns:
        Tensor: Boolean mask array containing indices of padded part.
    """
    in_shape = lengths.shape
    axis = len(in_shape) + 1 + axis if axis < 0 else axis
    prefix_shape = in_shape[:axis]
    suffix_shape = in_shape[axis:]
    lengths = jnp.expand_dims(lengths, axis=axis)  # (..., 1, ...)
    out_shape = prefix_shape + (maxlen,) + suffix_shape
    lengths = jnp.broadcast_to(lengths, out_shape)  # (..., maxlen, ...)
    mask = jnp.arange(maxlen)  # [0, 1, 2, ..., maxlen], (maxlen,)
    mask = jnp.expand_dims(mask, [1 + i for i in range(len(suffix_shape))])
    # (1, 1, ..., 1, maxlen, 1, ..., 1)
    mask = mask >= lengths
    return mask
