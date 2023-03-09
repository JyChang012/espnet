from typing import Optional, Tuple, Union

import flax.linen
import flax.linen as nn
import humanfriendly
import jax.numpy as jnp
import jax.random
from flax.linen import compact, merge_param
from jax import Array

from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnex.asr.frontend.abc import AbsFrontend
from espnex.layers.log_mel import LogMel
from espnex.layers.stft import Stft


class DefaultFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    fs: Union[int, str] = 16000
    win_length: Optional[int] = None
    hop_length: int = 128
    window: Optional[str] = "hann"
    center: bool = True
    normalized: bool = False
    onesided: bool = True
    n_mels: int = 80
    fmin: int = 0
    fmax: Optional[int] = None
    htk: bool = False
    frontend_conf: Optional[
        dict
    ] = None  # get_default_kwargs(Frontend)  # not implemented yet
    apply_stft: bool = True
    rng_collection: str = "channel"
    deterministic: Optional[bool] = None

    def output_size(self) -> int:
        return self.n_mels

    @compact
    def __call__(
        self, input: Array, input_lengths: Array, deterministic: Optional[bool] = None
    ) -> Tuple[Array, Array]:
        deterministic = merge_param("deterministic", self.deterministic, deterministic)

        if isinstance(self.fs, str):
            fs = humanfriendly.parse_size(self.fs)
        else:
            fs = self.fs

        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.apply_stft:
            input_stft, feats_lens = Stft(  # input_
                win_length=self.win_length,
                hop_length=self.hop_length,
                center=self.center,
                window=self.window,
                normalized=self.normalized,
                onesided=self.onesided,
            )(input, input_lengths)
        else:
            input_stft, feats_lens = input, input_lengths
        # input_stft is a complex array (B, T, Ch, F)

        # 2. [Option] Speech enhancement
        if self.frontend_conf is not None:
            raise NotImplementedError("Speech enhancement not supported yet!")

        # 3. [Multi channel case]: Select a channel
        if input_stft.ndim == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if not deterministic:
                # Select 1ch randomly
                ch = jax.random.randint(
                    self.make_rng(self.rng_collection), (), 0, input_stft.shape[2]
                )
                input_stft = input_stft[:, :, ch]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = LogMel(
            fs=fs,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            htk=self.htk,
        )(input_power, feats_lens)

        return input_feats, feats_lens
