from typing import Tuple, Optional

import librosa
import jax.numpy as jnp
import flax.linen as nn
from jax import Array

from espnex.models.utils import make_pad_mask


class LogMel(nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.mel

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
    """

    fs: int = 16000
    n_mels: int = 80
    fmin: float = 0.
    fmax: Optional[float] = None
    htk: bool = False
    log_base: Optional[float] = None

    @property
    def _mel_options(self):
        fmax = self.fs / 2 if self.fmax is None else self.fmax
        return dict(
            sr=self.fs,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=fmax,
            htk=self.htk,
        )

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self._mel_options.items())

    def __call__(
            self,
            feat: Array,
            ilens: Array = None,
    ) -> Tuple[Array, Array]:
        n_fft = (feat.shape[-1] - 1) * 2

        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        melmat = librosa.filters.mel(n_fft=n_fft, **self._mel_options).T

        mel_feat = feat @ melmat
        mel_feat = jnp.clip(mel_feat, a_min=1e-10)

        if self.log_base is None:
            logmel_feat = jnp.log(mel_feat)
        elif self.log_base == 2.0:
            logmel_feat = jnp.log2(mel_feat)
        elif self.log_base == 10.0:
            logmel_feat = jnp.log10(mel_feat)
        else:
            logmel_feat = jnp.log(mel_feat) / jnp.log(self.log_base)

        # Zero padding
        if ilens is not None:
            mask = make_pad_mask(jnp.expand_dims(ilens, -1), feat.shape[1], axis=1)  # (B, T, 1)
            logmel_feat = jnp.where(mask, 0., logmel_feat)
        else:
            ilens = None
        return logmel_feat, ilens
