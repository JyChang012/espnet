from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import Dense, Dropout, LayerNorm, Module, cond
from flax.linen.module import compact, merge_param
from jax import Array
from jax.random import bernoulli


class EncoderLayer(Module):
    """Encoder layer module.

    Args:
        self_attn (nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    self_attn: nn.Module
    feed_forward: nn.Module
    dropout_rate: float
    normalize_before: bool = True
    concat_after: bool = False
    stochastic_depth_rate: float = 0.0
    rng_collection: str = "skip_layer"
    deterministic: Optional[bool] = None

    @compact
    def __call__(
        self,
        x: Array,
        mask: Array,
        cache: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Array, Array]:
        deterministic = merge_param("deterministic", deterministic, self.deterministic)
        if deterministic or self.is_initializing() or self.stochastic_depth_rate == 0:
            return self.forward(x, mask, cache, deterministic)
        else:
            skip = bernoulli(
                self.make_rng(self.rng_collection), self.stochastic_depth_rate, ()
            )
            return cond(  # type: ignore
                skip,
                lambda mdl: (x, mask),
                lambda mdl: mdl.forward(x, mask, cache, deterministic),
                self,
            )

    def forward(
        self,
        x: Array,
        mask: Array,
        cache: Optional[jax.Array] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Array, Array]:
        """Compute encoded features.

        Args:
            x (jax.Array): Input tensor (#batch, time, size).
            mask (jax.Array): Mask tensor for the input (#batch, time).
            cache (jax.Array): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            jax.Array: Output tensor (#batch, time, size).
            jax.Array: Mask tensor (#batch, time).

        """
        stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)

        # currently ignore cache
        x_q = x

        if self.concat_after:
            x_concat = jnp.concatenate(
                (x, self.self_attn(x_q, x, mask, deterministic)), axis=-1
            )
            x = residual + stoch_layer_coeff * Dense(x.shape[-1])(x_concat)
        else:
            x = residual + stoch_layer_coeff * Dropout(self.dropout_rate)(
                self.self_attn(x_q, x, mask, deterministic), deterministic=deterministic
            )

        if not self.normalize_before:
            x = LayerNorm()(x)

        residual = x
        if self.normalize_before:
            x = LayerNorm()(x)
        x = residual + stoch_layer_coeff * Dropout(self.dropout_rate)(
            self.feed_forward(x), deterministic
        )
        if not self.normalize_before:
            x = LayerNorm()(x)

        # currently ignore cache

        return x, mask
