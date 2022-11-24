from typing import Optional

import flax.linen as nn
from flax.linen import Conv, Dense, Dropout, Module, merge_param, relu
from jax import Array


class MultiLayerConv1d(Module):
    """Multi-layered conv1d for Transformer block.

    This is a module of multi-leyered conv1d designed
    to replace positionwise feed-forward network
    in Transforner block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    hidden_channels: int
    kernel_size: int
    dropout_rate: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, deterministic: Optional[bool] = None) -> Array:
        """Calculate forward propagation.

        Args:
            x (Array): Batch of input tensors (B, T, in_chans).

        Returns:
            Array: Batch of output tensors (B, T, hidden_channels).

        """
        deterministic = merge_param("deterministic", self.deterministic, deterministic)
        in_channels = x.shape[-1]
        x = Conv(
            self.hidden_channels,
            self.kernel_size,
            strides=1,
            padding=(self.kernel_size - 1) // 2,
        )(x)
        x = relu(x)
        x = Dropout(self.dropout_rate)(x, deterministic)
        x = Conv(
            in_channels,
            self.kernel_size,
            strides=1,
            padding=(self.kernel_size - 1) // 2,
        )(x)
        return x


class Conv1dLinear(Module):
    """Conv1D + Linear for Transformer block.

    A variant of MultiLayeredConv1d, which replaces second conv-layer to linear.

    """

    hidden_channels: int
    kernel_size: int
    dropout_rate: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x: Array, deterministic: Optional[bool] = None) -> Array:
        """Calculate forward propagation.

        Args:
            x (Array): Batch of input tensors (B, T, in_chans).

        Returns:
            Array: Batch of output tensors (B, T, hidden_channels).

        """
        deterministic = merge_param("deterministic", self.deterministic, deterministic)
        in_channels = x.shape[-1]
        x = Conv(
            self.hidden_channels,
            self.kernel_size,
            strides=1,
            padding=(self.kernel_size - 1) // 2,
        )(x)
        x = relu(x)
        x = Dropout(self.dropout_rate)(x, deterministic)
        x = Dense(in_channels)(x)
        return x
