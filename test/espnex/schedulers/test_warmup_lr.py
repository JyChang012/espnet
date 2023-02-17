from espnex.schedulers import warmup_schedule
from optax import adam
from jax import random
import numpy as np
from jax.tree_util import tree_map
import jax
from pytest import mark


@mark.skip
def test_plot_lr():
    import matplotlib.pyplot as plt

    schedule = warmup_schedule(peak_lr=1e-3, warmup_steps=800)
    lrs = schedule(np.arange(5000))
    plt.plot(lrs)
    plt.show()


def test_warmup_schedule():
    def get_random_params():
        return dict(
            weight=np.random.uniform(-5, 5, [5, 10]),
            bias=np.random.uniform(-5, 5, [10])
        )

    params = get_random_params()
    schedule = warmup_schedule(peak_lr=1e-3)
    tx = adam(schedule)

    opt_state = tx.init(params)

    @jax.jit
    def update(grads, opt_state, params):
        updates, opt_state = tx.update(grads, opt_state, params)
        params = tree_map(lambda p, u: p + u, params, updates)
        return opt_state, params

    for _ in range(4):
        grads = get_random_params()
        opt_state, params = update(grads, opt_state, params)
