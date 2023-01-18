from typing import Union

from optax._src.base import Schedule
import jax.numpy as jnp


def warmup_schedule(
        peak_lr: float,
        warmup_steps: Union[int, float] = 25000
) -> Schedule:
    """The WarmupLR scheduler
    Linearly grow to peak_lr then inverse square root decay to zero as count go to infinity.
    lr = peak_lr * count / warmup_steps if count < warmup_steps else peak_lr * sqrt(warmup_steps / count)

    Args:
        peak_lr: peak learning rate
        warmup_steps: number of steps to warmup

    Returns: Schedule

    """

    def schedule(count):
        count += 1
        return peak_lr * warmup_steps ** .5 * jnp.minimum(
            count ** -.5, count * warmup_steps ** -1.5
        )
    return schedule



