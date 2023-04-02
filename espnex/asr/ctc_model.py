from typing import Sequence, Optional, Any, Tuple, Dict, Callable, Union, List
import logging

import numpy as np
from flax.linen import Dense
from jax import Array, tree_map
from optax import ctc_loss
import jax.numpy as jnp
from numpy import ndarray
from jax.nn.initializers import Initializer, glorot_uniform

from espnex.asr.encoder.abc import AbsEncoder
from espnex.asr.frontend.abc import AbsFrontend
from espnex.asr.utils import ASRErrorCalculator
from espnex.train.abs_espnex_model import AbsESPnetModel
from espnex.models.utils import make_pad_mask, inject_args
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnex.asr.ctc import ctc_decode

logger = logging.getLogger('ESPNex')


class CTCASRModel(AbsESPnetModel):
    vocab_size: int
    # token_list: Sequence[str]
    frontend: Optional[AbsFrontend]
    encoder: AbsEncoder
    ignore_id: int = -1
    # length_normalized_loss: bool = False
    # report_cer: bool = True
    # report_wer: bool = True
    blank_id: int = 0
    sym_space: str = "<space>"
    sym_blank: str = "<blank>"
    # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>
    # Pretrained HF Tokenizer needs custom sym_sos and sym_eo
    sym_sos: str = "<sos/eos>"
    sym_eos: str = "<sos/eos>"
    extract_feats_in_collect_stats: bool = True
    # lang_token_id: int = -1
    kernel_init: Optional[Initializer] = None

    def setup(self) -> None:
        dense = inject_args(Dense, kernel_init=self.kernel_init)
        self.ctc_out_dense = dense(self.vocab_size)

        if self.is_initializing():
            logger.info(str(self))

    def __call__(
            self,
            speech: Array,
            speech_lengths: Array,
            text: Array,
            text_lengths: Array,
            training: bool,
            *args: Any,
            **kwargs: Any,
    ) -> Tuple[Array, Dict[str, Any], float, Tuple[Array, Array, Array, Array]]:
        batch_size = (speech_lengths > 0) | (text_lengths > 0)
        batch_size = jnp.sum(batch_size)

        enc_out, enc_out_lengths = self.encode(speech, speech_lengths, training)
        enc_out = self.ctc_out_dense(enc_out)
        enc_padded_mask = make_pad_mask(enc_out_lengths, enc_out.shape[1])
        text_padded_mask = make_pad_mask(text_lengths, text.shape[1])

        losses = ctc_loss(enc_out,
                          enc_padded_mask,
                          text,
                          text_padded_mask,
                          blank_id=self.blank_id)  # (bs,)
        loss = jnp.sum(losses) / batch_size

        arg_max_enc_out = jnp.argmax(enc_out, axis=-1)
        decoded, decoded_length = ctc_decode(arg_max_enc_out, enc_out_lengths, self.blank_id, self.ignore_id)

        return loss, {'loss': loss}, batch_size, (decoded, decoded_length, text, text_lengths)

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

    def build_evaluator(
            self,
            token_list: Union[Tuple[str, ...], List[str]]
    ) -> Callable[[float, Dict[str, Any], float, Any], Dict[str, Any]]:
        # error_calculator use -1 as padding idx
        ec = ASRErrorCalculator(
            token_list.copy(), space_sym=self.sym_space, blank_id=self.blank_id
        )

        # ec = ErrorCalculator(token_list, self.sym_space, self.sym_blank, True, True)
        ec = ErrorCalculator(token_list, '<space>', self.sym_blank, True, True)
        # for reproducing original result, we hardcode sym_space here like old ESPNet

        def evaluate(
                loss: float,
                stats: Dict[str, Any],
                weight: float,
                aux: Tuple[ndarray, ndarray, ndarray, ndarray],

                # input to __call__
                speech: Array,
                speech_lengths: Array,
                text: Array,
                text_lengths: Array,
                return_decoded: Optional[bool] = False
        ) -> Dict[str, Any]:

            def truncate(arr):
                return arr[:weight]

            aux = tree_map(truncate, aux)

            decoded, decoded_length, text, text_lengths = aux


            decoded_str = ec.batch_tokens2str(decoded, decoded_length)
            text_str = ec.batch_tokens2str(text, text_lengths)

            cer = ec.calculate_cer(decoded_str, text_str)
            wer = ec.calculate_wer(decoded_str, text_str)

            stats.update(dict(
                wer=wer,
                cer=cer,
            ))
            if return_decoded:
                return stats, dict(decoded_str=decoded_str,
                                   text_str=text_str)

            return stats
        return evaluate








