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
from espnex.models.utils import inject_args, make_pad_mask, shift_right, shift_left
from espnex.models.transformer.decoder_layer import DecoderLayer
from espnex.models.transformer.embedding import AddPositionalEncoding
from espnex.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnex.models.transformer.stochastic_sequential import StochasticSequential
from espnex.asr.decoder.abc import AbsDecoder
from espnex.train.abs_espnex_model import AbsESPnetModel
from espnex.models.loss import LabelSmoothingLoss


class ESPnetASRModelOutputAux(PyTreeNode):
    ctc_decoded: Optional[Array]
    ctc_decoded_lens: Optional[Array]
    attention_decoded: Optional[Array]  # same length as ground truth
    targets: Array
    targets_lens: Array


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
    lang_token_id: int = -1

    ctc_weight: float = 0.5
    lsm_weight: float = 0.0

    sym_space: str = "<space>"
    sym_blank: str = "<blank>"
    # In a regular ESPnet recipe, <sos> and <eos> are both "<sos/eos>"
    # Pretrained HF Tokenizer needs custom sym_sos and sym_eos
    sym_sos: str = "<sos/eos>"
    sym_eos: str = "<sos/eos>"

    length_normalized_loss: bool = False
    report_cer: bool = True
    report_wer: bool = True
    extract_feats_in_collect_stats: bool = True
    kernel_init: Optional[Initializer] = None

    def setup(self) -> None:
        assert 0 <= self.ctc_weight <= 1., 'CTC weight needs to be within [0, 1]!'
        if self.ctc_weight < 1.:
            assert self.decoder is not None, "Decoder should not be None when attention is used"
        else:
            self.decoder = None
            logging.warning("Set decoder to none as ctc_weight==1.0")
        reduction = 'mean' if self.length_normalized_loss else 'sum'
        self.criterion_att = LabelSmoothingLoss(smoothing=self.lsm_weight, reduction=reduction)

        dense = inject_args(Dense, kernel_init=self.kernel_init)
        self.ctc_out_dense = dense(self.vocab_size)

    def __call__(self,
                 speech: Array,
                 speech_lengths: Array,
                 text: Array,
                 text_lengths: Array,
                 training: bool,
                 **kwargs) -> Tuple[Array, Dict[str, Any], float, ESPnetASRModelOutputAux]:
        enc_out, enc_out_lengths = self.encode(speech, speech_lengths, training)
        enc_padded_mask = make_pad_mask(enc_out_lengths, enc_out.shape[1])
        text_padded_mask = make_pad_mask(text_lengths, text.shape[1])
        stats = dict()
        bsize = (text_lengths > 0) | (speech_lengths > 0)
        bsize = jnp.sum(bsize)

        loss_ctc = loss_att = 0
        ctc_decoded = ctc_decoded_lengths = att_decoded = None
        # 1. CTC branch
        # TODO: Note: we assume that there is an <sos> / <space> at the start of each `text`
        if self.ctc_weight != 0.:
            loss_ctc = ctc_loss(enc_out,
                                enc_padded_mask,
                                text[:, 1:],
                                text_padded_mask[:, 1:],
                                blank_id=self.blank_id)  # (bs,)
            loss_ctc = jnp.sum(loss_ctc) / bsize

            arg_max_enc_out = jnp.argmax(enc_out, axis=-1)
            ctc_decoded, ctc_decoded_lengths = ctc_decode(arg_max_enc_out,
                                                          enc_out_lengths,
                                                          self.blank_id,
                                                          self.ignore_id)
            stats['loss_ctc'] = loss_ctc

        # 2. Attention decoder branch
        if self.ctc_weight != 1.:
            loss_att, acc_att, att_decoded = self.calculate_decoder_loss(enc_out,
                                                                         enc_out_lengths,
                                                                         text,
                                                                         text_lengths,
                                                                         training)
            stats['loss_att'] = loss_att
            stats['acc'] = acc_att

        loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
        stats['loss'] = loss
        return loss, stats, bsize, ESPnetASRModelOutputAux(ctc_decoded,
                                                           ctc_decoded_lengths,
                                                           att_decoded,
                                                           text[:, 1:],
                                                           text_lengths - 1)  # ignore <eos>

    def calculate_decoder_loss(self,
                               encoder_out: Array,
                               encoder_out_lengths: Array,
                               targets: Array,
                               targets_lengths: Array,
                               training: bool):
        # TODO: Note: we assume here that there is an <sos> at the start of each `text`
        inputs = targets.at[:, 0].set(self.sos_id)
        targets = shift_left(targets, self.ignore_id)
        targets = targets.at[list(range(targets.shape[0])), targets_lengths - 1].set(self.eos_id)
        batch_size = (encoder_out_lengths > 0) | (targets_lengths > 0)
        batch_size = jnp.sum(batch_size)

        # 1. forward decoder
        # TODO: support weight tying
        decoder_out, _ = self.decoder(inputs,
                                      targets_lengths,
                                      encoder_out,
                                      encoder_out_lengths,
                                      not training,
                                      decode=False)
        targets_mask = ~make_pad_mask(targets_lengths, targets.shape[1])
        targets_mask = targets_mask.astype(int)

        loss_att = self.criterion_att(jax.nn.log_softmax(decoder_out),
                                      targets,
                                      weights=targets_mask)
        if not self.length_normalized_loss:
            loss_att = loss_att / batch_size
        att_decoded = jnp.argmax(decoder_out, axis=-1) * targets_mask
        acc_att = att_decoded == targets
        acc_att = jnp.sum(acc_att * targets_mask) / jnp.sum(targets_mask)

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
            logging.warning(
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
    ) -> Callable[[float, Dict[str, Any], float, ESPnetASRModelOutputAux], Dict[str, Any]]:
        # error_calculator use -1 as padding idx
        error_calculator = ErrorCalculator(
            token_list.copy(), self.sym_space, self.sym_blank, self.report_cer, self.report_wer
        )

        def convert2char(arr, arr_len):
            ret = []
            for x, xlen in zip(arr, arr_len):
                x = x[:xlen]
                ret.append(''.join(map(token_list.__getitem__, x)))
            return ret

        def evaluate(
                loss: float,
                stats: Dict[str, Any],
                weight: float,
                aux: ESPnetASRModelOutputAux,
                return_decoded: Optional[bool] = False
        ) -> Dict[str, Any]:

            bsize = weight
            aux = tree_map(lambda arr: arr[:bsize], aux)

            (ctc_decoded,
             ctc_decoded_lens,
             att_decoded,
             text,
             text_lengths) = (aux.ctc_decoded,
                              aux.ctc_decoded_lens,
                              aux.attention_decoded,
                              aux.targets,
                              aux.targets_lens)

            ctc_decoded = ctc_decoded[:, :np.max(ctc_decoded_lens)]

            text_maxlen = np.max(text_lengths)
            text = text[:, :text_maxlen]
            att_decoded = att_decoded[:, :text_maxlen]

            ctc_decoded_str = convert2char(ctc_decoded, ctc_decoded_lens)
            att_decoded_str = convert2char(att_decoded, text_lengths)
            text_str = convert2char(text, text_lengths)

            '''
            # filter out empty text_str
            att_decoded_str, ctc_decoded_str, text_str = zip(
                *((a, d, t) for a, d, t in zip(att_decoded_str, ctc_decoded_str, text_str) if t)
            )
            '''

            cer_ctc = error_calculator.calculate_cer(ctc_decoded_str, text_str)
            wer_ctc = error_calculator.calculate_wer(ctc_decoded_str, text_str)
            wer = error_calculator.calculate_wer(att_decoded_str, text_str)
            cer = error_calculator.calculate_cer(att_decoded_str, text_str)

            stats.update(dict(
                wer_ctc=wer_ctc,
                cer_ctc=cer_ctc,
                wer=wer,
                cer=cer
            ))
            if return_decoded:
                # return stats, ctc_decoded_str, att_decoded_str, text_str
                return stats, dict(text_str=text_str,
                                   ctc_decoded_str=ctc_decoded_str,
                                   att_decoded_str=att_decoded_str)

            return stats

        return evaluate
