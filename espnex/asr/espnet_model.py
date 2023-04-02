from functools import partial
from typing import Callable, Optional, Tuple, Union, Type, Literal, Any, Dict, Sequence, List
import logging

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen import Dense, Dropout, LayerNorm, Module, cond, MultiHeadDotProductAttention, compact, Sequential, Embed
from flax.linen.module import compact, merge_param
from flax.linen.activation import relu
from jax import Array, tree_map
from jax.nn.initializers import Initializer, glorot_uniform
from flax.linen.attention import make_causal_mask, combine_masks, make_attention_mask
from numpy import ndarray
from optax import ctc_loss
from flax.struct import PyTreeNode, field

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnex.asr.ctc import ctc_decode
from espnex.asr.encoder.abc import AbsEncoder
from espnex.asr.frontend.abc import AbsFrontend
from espnex.asr.utils import ASRErrorCalculator
from espnex.models.utils import inject_args, make_pad_mask, shift_right, shift_left
from espnex.models.transformer.decoder_layer import DecoderLayer
from espnex.models.transformer.embedding import AddPositionalEncoding
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.stochastic_sequential import StochasticSequential
from espnex.asr.decoder.abc import AbsDecoder
from espnex.train.abs_espnex_model import AbsESPnetModel
from espnex.models.loss import LabelSmoothingLoss

logger = logging.getLogger('ESPNex')


class ESPnetASRModelOutputAux(PyTreeNode):
    ctc_output: Optional[Array]
    ctc_output_lengths: Optional[Array]
    attention_decoded: Optional[Array]  # same length as ground truth


class ESPnetASRModel(AbsESPnetModel):
    vocab_size: int

    # token_list: Sequence[str]
    frontend: Optional[AbsFrontend]
    encoder: AbsEncoder
    decoder: Optional[AbsDecoder]

    sos_id: int
    eos_id: int
    ignore_id: int = -1
    blank_id: int = 0
    # lang_token_id: int = -1

    ctc_weight: float = 0.5
    lsm_weight: float = 0.0

    sym_space: str = "<space>"
    sym_blank: str = "<blank>"
    # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
    # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
    sym_sos: str = "<sos/eos>"
    sym_eos: str = "<sos/eos>"

    length_normalized_loss: bool = False
    # report_cer: bool = True
    # report_wer: bool = True
    extract_feats_in_collect_stats: bool = True
    kernel_init: Optional[Initializer] = None

    def setup(self) -> None:
        assert 0 <= self.ctc_weight <= 1., 'CTC weight needs to be within [0, 1]!'
        if self.ctc_weight < 1.:
            assert self.decoder is not None, "Decoder should not be None when attention is used"
        elif self.is_initializing() and self.decoder is not None:
            logger.warning(
                "ctc_weight == 1.0 but self.decoder is not None. Weights of decoder will not be initialized!"
            )
        reduction = 'mean' if self.length_normalized_loss else 'sum'
        self.criterion_att = LabelSmoothingLoss(smoothing=self.lsm_weight, reduction=reduction)

        dense = inject_args(Dense, kernel_init=self.kernel_init)
        self.ctc_out_dense = dense(self.vocab_size)

        if self.is_initializing():
            logger.info(str(self))

    def __call__(self,
                 speech: Array,
                 speech_lengths: Array,
                 text: Array,
                 text_lengths: Array,
                 training: bool,
                 **kwargs) -> Tuple[Array, Dict[str, Any], float, ESPnetASRModelOutputAux]:
        batch_size = (speech_lengths > 0) | (text_lengths > 0)
        batch_size = jnp.sum(batch_size)

        enc_out, enc_out_lengths = self.encode(speech, speech_lengths, training)
        enc_padded_mask = make_pad_mask(enc_out_lengths, enc_out.shape[1])
        text_padded_mask = make_pad_mask(text_lengths, text.shape[1])
        stats = dict()

        loss_ctc = loss_att = 0
        ctc_output = att_decoded = None
        # 1. CTC branch
        if self.ctc_weight != 0:
            ctc_logits = self.ctc_out_dense(enc_out)
            loss_ctc = ctc_loss(ctc_logits,  # TODO: test
                                enc_padded_mask,
                                text,
                                text_padded_mask,
                                blank_id=self.blank_id)  # (bs,)
            loss_ctc = jnp.sum(loss_ctc) / batch_size

            ctc_output = jnp.argmax(ctc_logits, axis=-1)
            # ctc_output = jnp.where(enc_padded_mask, self.ignore_id, ctc_output)
            # ctc_decoded, ctc_decoded_lengths = ctc_decode(arg_max_ctc_logits,
            #                                               enc_out_lengths,
            #                                               self.blank_id,
            #                                               self.ignore_id)
            stats['loss_ctc'] = loss_ctc

        # 2. Attention decoder branch
        if self.ctc_weight != 1:
            loss_att, acc_att, att_decoded = self.calculate_decoder_loss(enc_out,
                                                                         enc_out_lengths,
                                                                         text,
                                                                         text_lengths,
                                                                         training)
            stats['loss_att'] = loss_att
            stats['acc'] = acc_att

        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        stats['loss'] = loss
        return loss, stats, batch_size, ESPnetASRModelOutputAux(ctc_output, enc_out_lengths, att_decoded)

    def calculate_decoder_loss(self,
                               encoder_out: Array,
                               encoder_out_lengths: Array,
                               targets: Array,
                               targets_lengths: Array,
                               training: bool):
        # assume a position is reserved for <sos> / <eos>, to deal with the static shape requirement of XLA compiler
        inputs = shift_right(targets, self.sos_id)
        targets = targets.at[list(range(targets.shape[0])), targets_lengths].set(self.eos_id)

        batch_size = (encoder_out_lengths > 0) | (targets_lengths > 0)
        batch_size = jnp.sum(batch_size)

        targets_lengths = jnp.where(targets_lengths > 0, targets_lengths + 1, 0)

        # 1. forward decoder
        decoder_out, _ = self.decoder(inputs,
                                      targets_lengths,
                                      encoder_out,
                                      encoder_out_lengths,
                                      not training,
                                      decode=False)
        targets_mask = ~make_pad_mask(targets_lengths, targets.shape[1])

        loss_att = self.criterion_att(jax.nn.log_softmax(decoder_out),
                                      targets,
                                      weights=targets_mask)
        if not self.length_normalized_loss:
            loss_att = loss_att / batch_size
        att_decoded = jnp.argmax(decoder_out, axis=-1) * targets_mask
        acc_att = att_decoded == targets
        acc_att = jnp.where(targets_mask, acc_att, 0)
        acc_att = jnp.sum(acc_att) / jnp.sum(targets_mask)

        return loss_att, acc_att, att_decoded

    def _extract_feats(self,
                       speech: Array,
                       speech_lengths: Array,
                       training: bool):
        if self.frontend is not None:
            speech, speech_lengths = self.frontend(speech, speech_lengths, not training)
        return speech, speech_lengths

    def collect_feats(self,
                      speech: Array,
                      speech_lengths: Array,
                      text: Optional[Array],
                      text_lengths: Optional[Array],
                      training: bool,
                      *args: Any,
                      **kwargs: Any) -> Dict[str, Array]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths, training)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logger.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(self,
               speech: Array,
               speech_lengths: Array,
               training: bool):
        """Frontend + encoder"""
        feats, feats_lengths = self._extract_feats(speech, speech_lengths, training)
        enc_out, enc_out_lengths, _ = self.encoder(feats, feats_lengths, deterministic=not training)
        return enc_out, enc_out_lengths

    def build_evaluator(
            self,
            token_list: Union[Tuple[str, ...], List[str]]
    ) -> Callable[..., Dict[str, Any]]:
        # error_calculator use -1 as padding idx
        # error_calculator = ErrorCalculator(
        #     token_list.copy(), self.sym_space, self.sym_blank, self.report_cer, self.report_wer
        # )

        ec = ASRErrorCalculator(token_list.copy(), blank_id=self.blank_id, space_sym=self.sym_space)

        def evaluate(
                loss: float,
                stats: Dict[str, Any],
                weight: float,
                aux: ESPnetASRModelOutputAux,

                # input to __call__
                speech: Array,
                speech_lengths: Array,
                text: Array,
                text_lengths: Array,
                return_decoded: Optional[bool] = False
        ) -> Dict[str, Any]:

            bsize = weight
            aux, text, text_lengths = tree_map(
                lambda arr: arr[:bsize], (aux, text, text_lengths)
            )
            ctc_output, ctc_out_lengths, att_decoded = aux.ctc_output, aux.ctc_output_lengths, aux.attention_decoded

            text_str = ec.batch_tokens2str(text, text_lengths)

            decoded = dict(text_str=text_str)
            if ctc_output is not None:
                ctc_str = ec.batch_ctc_decode_str(ctc_output, ctc_out_lengths)
                decoded['ctc_str'] = ctc_str

                # tmp = tuple(map(lambda arr: arr[1:], text_str))

                stats['cer_ctc'] = ec.calculate_cer(ctc_str, text_str)
                stats['wer_ctc'] = ec.calculate_wer(ctc_str, text_str)

            if att_decoded is not None:
                att_str = ec.batch_tokens2str(att_decoded, text_lengths)
                decoded['att_str'] = att_str
                stats['cer'] = ec.calculate_cer(att_str, text_str)
                stats['wer'] = ec.calculate_wer(att_str, text_str)

            if return_decoded:
                decoded['ctc_original'] = ec.batch_tokens2str(ctc_output, ctc_out_lengths)
                # return stats, ctc_decoded_str, att_decoded_str, text_str
                return stats, decoded
            return stats
        return evaluate
