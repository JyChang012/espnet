from typing import Tuple

import jax.numpy as jnp
from jax import Array
from jax import lax
from flax.struct import PyTreeNode


# Jittable CTC decode function, might be an overkill here
class CTCDecodeState(PyTreeNode):
    output: Array  # output of the CTC decoder, of the same size as input due to limitation of XLA compiler
    output_positions: Array  # Current position to update of each row. It's also the lengths of decoded output.
    input_position: Array = jnp.array(0)  # scalar int array indicating the position in input


def ctc_decode(
        x: Array,
        x_lengths: Array,
        blank_id: int = 0,
        padding_id: int = -1,
) -> Tuple[Array, Array]:
    """
    CTC decode function.
    Args:
        x: (Batch, maxlen) input to decode
        x_lengths: (Batch,) lengthds of input
        blank_id: int blank id for CTC
        padding_id: int: token used for padding in output, should not be in the vocabulary

    Returns:
        output: decoded output, same shape as x
        output_lengths: (Batch,) lengths of each example in batch

    """
    logits_shifted_right = jnp.roll(x, 1, axis=1)
    logits_shifted_right = logits_shifted_right.at[:, 0].set(padding_id)
    eql = logits_shifted_right == x
    x = jnp.where(eql, padding_id, x)

    batch_size, max_len, *_ = x.shape

    def cond_fn(state: CTCDecodeState) -> Array:  # return a scalar boolean array
        return jnp.any(state.input_position < x_lengths)

    def update_state(state: CTCDecodeState) -> CTCDecodeState:
        cur_x = x[:, state.input_position]
        updating_positions = (cur_x != padding_id) & (cur_x != blank_id) & (state.input_position < x_lengths)
        update = jnp.where(updating_positions, cur_x, padding_id)
        output = state.output.at[jnp.arange(batch_size), state.output_positions].set(update)
        return CTCDecodeState(
            output,
            state.output_positions + updating_positions,
            state.input_position + 1,
            )

    state = CTCDecodeState(
        jnp.full_like(x, padding_id, dtype=int),
        jnp.zeros((batch_size,), dtype=int),
    )
    state = lax.while_loop(cond_fn, update_state, state)
    return state.output, state.output_positions
