from functools import partial
from typing import Optional, Callable, Any
from inspect import signature

import jax.numpy as jnp
from jax import Array


def make_pad_mask(  # a simplified version of the torch one, support JIT compilation.
        lengths: Array, maxlen: int, axis: int = -1
) -> Array:
    """Make mask tensor containing indices of padded part. Padded part is 1, others are zero. An additional dimension of
     length `maxlen` is inserted at `axis` position, indicating padding.

    Examples:
        lengths is of shape (4, 5, 3), axis is 1, maxlen is 200, then output is of shape (4, 200, 5, 3).

    Args:
        lengths (LongTensor or List): Batches of lengths (*B,)
        maxlen (int): Length of the returned mask array
        axis: position to insert the padding axis

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
    mask = jnp.expand_dims(mask, [1 + i for i in range(len(suffix_shape))])  # (maxlen, ...)
    # (1, 1, ..., 1, maxlen, 1, ..., 1)
    mask = mask >= lengths  # (maxlen, ...) >= (..., 1, ...) -> (..., maxlen, ...)
    return mask


def shift_right(x: Array,
                value: float,
                axis: int = 1):
    """Shift the input to the right by padding on axis 1."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(value))
    return padded[:, :-1]


def shift_left(x: Array,
                value: float,
                axis: int = 1):
    """Shift the input to the right by padding on axis 1."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (0, 1)
    padded = jnp.pad(
        x, pad_widths, mode='constant', constant_values=x.dtype.type(value))
    return padded[:, 1:]


def inject_args(f: Callable,
                *args: Any,
                filter_none: bool = True,
                inspect: bool = True,
                **kwargs: Any) -> Callable:
    """
    Similar to `functools.partial`, but with additional checks on parameters.
    """
    sig = signature(f)
    kwargs = {k: v for k, v in kwargs.items()
              if ((not filter_none) or v is not None)
              and
              ((not inspect) or k in sig.parameters)}
    return partial(f, *args, **kwargs) if kwargs or args else f
