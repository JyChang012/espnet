from typing import Optional

import jax.numpy as jnp
from jax import Array


def make_pad_mask(  # a simplified version of the torch one, to support JIT compilation.
        lengths: Array,
        maxlen: int,
) -> Array:
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batches of lengths (*B,).
        maxlen (int): Length of the returned mask array


    Returns:
        Tensor: Boolean mask array containing indices of padded part.
    """
    bs = lengths.shape
    mask = jnp.arange(maxlen)
    mask = jnp.broadcast_to(mask, [*bs, maxlen])  # (*bs, max_len)
    lengths = jnp.expand_dims(lengths, -1)
    mask = mask >= lengths  # (*bs, max_len)
    return mask



