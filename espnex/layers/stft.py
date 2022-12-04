from typing import Optional, Tuple, Union

import jax.numpy as jnp
from flax.linen import Module
from jax import Array
from jax.scipy.signal import stft

from espnex.layers.abc import InversibleInterface
from espnex.models.transformer.subsampling import get_output_lengths
from espnex.models.utils import make_pad_mask


class Stft(InversibleInterface, Module):
    n_fft: int = 512
    win_length: Optional[int] = None
    hop_length: int = 128
    window: Optional[str] = "hann"
    center: bool = True
    pad_at_end: bool = False  # additional argument for scipy's stft
    normalized: bool = False
    onesided: bool = True

    def extra_repr(self):
        return (
            f"n_fft={self.n_fft}, "
            f"win_length={self.win_length}, "
            f"hop_length={self.hop_length}, "
            f"center={self.center}, "
            f"normalized={self.normalized}, "
            f"onesided={self.onesided}"
        )

    def __call__(
            self,
            input: Array,
            ilens: Optional[Array] = None
    ) -> Tuple[Array, Optional[Array]]:
        """STFT forward function.

        Args:
            input: (Batch, Nsamples) or (Batch, Nsample, Channels)
            ilens: (Batch)
        Returns:
            output: (Batch, Frames, Freq) or (Batch, Frames, Channels, Freq)

        """

        win_length = self.win_length if self.win_length is not None else self.n_fft
        noverlap = win_length - self.hop_length
        *_, output = stft(
            input,
            window=self.window,
            nperseg=win_length,
            noverlap=noverlap,
            nfft=self.n_fft,
            return_onesided=self.onesided,
            boundary=None if not self.center else 'zeros',  # does not support 'reflect' used by torch.stft by default
            padded=self.pad_at_end,
            axis=1,  # Nsamples dimension
        )
        output: Array
        if self.normalized:
            output = output * (win_length ** -.5)

        if output.ndim == 4:
            with_channel = True
            # bs, freq, channel, frames
            output = jnp.transpose(output, [0, 3, 2, 1])  # bs, frames, channel, freq
        else:
            with_channel = False
            # bs, freq, frames
            output = jnp.transpose(output, [0, 2, 1])  # bs, frames, freq

        if ilens is not None:
            pad = win_length // 2 if self.center else 0
            olens = get_output_lengths(ilens, win_length, self.hop_length, pad)

            mask = make_pad_mask(olens, output.shape[1])  # (bs, frames)
            mask = jnp.expand_dims(
                mask,
                [2, 3] if with_channel else 2
            )  # (bs, frames, 1)
            output = jnp.where(mask, 0., output)
        else:
            olens = None

        output = output * (self.n_fft / 2)  # ensure same result as torch.stft

        return output, olens

    def inverse(
            self,
            input: Array,
            ilens: Array = None
    ) -> Tuple[Array, Optional[Array]]:
        raise NotImplementedError  # TODO: implement inverse
