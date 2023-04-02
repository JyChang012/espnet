from typing import Optional, Callable, Literal

import jax.numpy as jnp
import numpy as np
from jax import Array
import jax
import chex
from optax import smooth_labels

# TODO: remove `kl_divergence` after the next release of optax
def kl_divergence(
        log_predictions: chex.Array, targets: chex.Array
) -> chex.Array:
    """Computes the Kullback-Leibler divergence (relative entropy) loss.
    Measures the information gain achieved if target probability distribution
    would be used instead of predicted probability distribution.
    References:
      [Kullback, Leibler, 1951](https://www.jstor.org/stable/2236703)
    Args:
      log_predictions: Probabilities of predicted distribution with shape [...,
        dim]. Expected to be in the log-space to avoid underflow.
      targets: Probabilities of target distribution with shape [..., dim].
        Expected to be strictly positive.
    Returns:
      Kullback-Leibler divergence of predicted distribution from target
      distribution with shape [...].
    """
    chex.assert_type([log_predictions, targets], float)
    loss = targets * (
            jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions
    )
    return jnp.sum(loss, axis=-1)


def LabelSmoothingLoss(smoothing: float = 0.,
                       criterion: Callable[[Array, Array], Array] = kl_divergence,
                       reduction: Optional[Literal['mean', 'sum']] = None):
    """
    Args:
        smoothing:
        criterion:
        reduction:

    Returns:

    """
    def label_smoothing_loss(log_predictions: Array,
                             targets: Array,
                             weights: Optional[Array] = None):
        """
        Args:
            log_predictions: [*bs, n_classes]
            targets: [*bs]
            weights: [*bs] can also be used as mask

        Returns:

        """
        n_class = log_predictions.shape[-1]
        assert n_class > 1, '# of classes must be larger than 1!'
        alpha = smoothing / (n_class - 1) * n_class
        smoothed_targets = smooth_labels(jax.nn.one_hot(targets, n_class), alpha)
        ret = criterion(log_predictions, smoothed_targets)
        if weights is not None:
            ret = ret * weights
        if reduction == 'sum':
            ret = jnp.sum(ret)
        elif reduction == 'mean':
            n_tokens = jnp.sum(weights) if weights is not None else ret.size
            ret = jnp.sum(ret) / n_tokens
        return ret

    return label_smoothing_loss

