from typing import Sequence, Optional, Any, Tuple, Dict
import logging

import flax.linen as nn
from flax.linen import Module, Dense
from jax import Array
from flax.struct import PyTreeNode
from optax import ctc_loss
import jax.numpy as jnp

from espnex.asr.encoder.abc import AbsEncoder
from espnex.asr.frontend.abc import AbsFrontend
from espnex.train.abs_espnex_model import AbsESPnetModel
from espnex.models.utils import make_pad_mask


class CTCASRModel(AbsESPnetModel):
    vocab_size: int
    # token_list: Sequence[str]
    frontend: Optional[AbsFrontend]
    encoder: AbsEncoder
    ignore_id: int = -1
    length_normalized_loss: bool = False
    report_cer: bool = True
    report_wer: bool = True
    sym_space: str = "<space>"
    sym_blank: str = "<blank>"
    # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>
    # Pretrained HF Tokenizer needs custom sym_sos and sym_eo
    sym_sos: str = "<sos/eos>"
    sym_eos: str = "<sos/eos>"
    extract_feats_in_collect_stats: bool = True
    lang_token_id: int = -1

    def setup(self) -> None:
        self.out_dense = Dense(self.vocab_size)

    def __call__(
            self,
            speech: Array,
            speech_lengths: Array,
            text: Array,
            text_lengths: Array,
            training: bool,
            *args: Any,
            **kwargs: Any,
    ) -> Tuple[Array, int, Any]:
        enc_out, enc_out_lengths = self.encode(speech, speech_lengths, training)
        enc_out = self.out_dense(enc_out)
        enc_padded_mask = make_pad_mask(enc_out_lengths, enc_out.shape[1])
        # enc_padded_mask = enc_padded_mask.astype('float')
        text_padded_mask = make_pad_mask(text_lengths, text.shape[1])
        losses = ctc_loss(enc_out, enc_padded_mask, text, text_padded_mask)  # (bs,)
        loss = jnp.mean(losses)
        batch_size = speech.shape[0]
        return loss, batch_size, (enc_out, enc_out_lengths)

    def _extract_feats(
            self,
            speech: Array,
            speech_lengths: Array,
            training: bool
    ):
        if self.frontend is not None:
            speech, speech_lengths = self.frontend(speech, speech_lengths, not training)
        return speech, speech_lengths

    def collect_feats(
            self,
            speech: Array,
            speech_lengths: Array,
            text: Optional[Array],
            text_lengths: Optional[Array],
            training: bool,
            *args: Any,
            **kwargs: Any,
    ) -> Dict[str, Array]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths, training)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
            self,
            speech: Array,
            speech_lengths: Array,
            training: bool
    ):
        """Frontend + encoder"""
        feats, feats_lengths = self._extract_feats(speech, speech_lengths, training)
        enc_out, enc_out_lengths, _ = self.encoder(feats, feats_lengths, deterministic=not training)
        return enc_out, enc_out_lengths
