import argparse
import logging
from typing import Callable, Collection, Dict, List, Optional, Tuple, Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jax import Array
from typeguard import check_argument_types, check_return_type

from espnet2.train.class_choices import ClassChoices
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.utils.types import float_or_none, int_or_none, str2bool, str_or_none
from espnex.asr.ctc_model import CTCASRModel
from espnex.asr.encoder.abc import AbsEncoder
from espnex.asr.encoder.transformer_encoder import TransformerEncoder
from espnex.asr.frontend.abc import AbsFrontend
from espnex.asr.frontend.default import DefaultFrontend
from espnex.tasks.abc import AbsTask
from espnex.train.abs_espnex_model import AbsESPnetModel
from espnex.train.collate_fn import CommonCollateFn
from espnet2.tasks.asr import preprocessor_choices


frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)

model_choices = ClassChoices(
    "model",
    classes=dict(
        ctcasr=CTCASRModel,
    ),
    type_check=AbsESPnetModel,
    default="ctcasr",
)

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        transformer=TransformerEncoder,
    ),
    type_check=AbsEncoder,
    default="transformer",
)



class ASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --model and --model_conf
        model_choices,
        # --preencoder and --preencoder_conf
        # --encoder and --encoder_conf
        encoder_choices,
        # --preprocessor and --preprocessor_conf
        preprocessor_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    # trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        # group.add_argument(
        #     "--ctc_conf",
        #     action=NestedDictAction,
        #     default=get_default_kwargs(CTC),
        #     help="The keyword arguments for CTC class.",
        # )
        # group.add_argument(
        #     "--joint_net_conf",
        #     action=NestedDictAction,
        #     default=None,
        #     help="The keyword arguments for joint network class.",
        # )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn", "hugging_face"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        group.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        group.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        group.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        group.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        group.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        group.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        group.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )
        group.add_argument(
            "--short_noise_thres",
            type=float,
            default=0.5,
            help="If len(noise) / len(speech) is smaller than this threshold during "
            "dynamic mixing, a warning will be displayed.",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, Array]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:

            try:
                _ = getattr(args, "preprocessor")
            except AttributeError:
                setattr(args, "preprocessor", "default")
                setattr(args, "preprocessor_conf", dict())
            except Exception as e:
                raise e

            preprocessor_class = preprocessor_choices.get_class(args.preprocessor)
            retval = preprocessor_class(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                # NOTE(kamo): Check attribute existence for backward compatibility
                rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                rir_apply_prob=args.rir_apply_prob
                if hasattr(args, "rir_apply_prob")
                else 1.0,
                noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                noise_apply_prob=args.noise_apply_prob
                if hasattr(args, "noise_apply_prob")
                else 1.0,
                noise_db_range=args.noise_db_range
                if hasattr(args, "noise_db_range")
                else "13_15",
                short_noise_thres=args.short_noise_thres
                if hasattr(args, "short_noise_thres")
                else 0.5,
                speech_volume_normalize=args.speech_volume_normalize
                if hasattr(args, "rir_scp")
                else None,
                **args.preprocessor_conf,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        MAX_REFERENCE_NUM = 4
        retval = []
        retval += ["text_spk{}".format(n) for n in range(2, MAX_REFERENCE_NUM + 1)]
        retval = tuple(retval)
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(
            cls, args: argparse.Namespace
    ) -> Tuple[AbsESPnetModel, FrozenDict, str, Callable[..., Dict[str, Any]]]:
        assert check_argument_types()

        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram

        # 3. Normalization layer

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(**args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        # encoder_output_size = encoder.output_size()
        # if getattr(args, "postencoder", None) is not None:
        #     postencoder_class = postencoder_choices.get_class(args.postencoder)
        #     postencoder = postencoder_class(
        #         input_size=encoder_output_size, **args.postencoder_conf
        #     )
        #     encoder_output_size = postencoder.output_size()
        # else:
        #     postencoder = None

        # 5. Decoder
        # 7. Build model
        # try:
        #     model_class = model_choices.get_class(args.model)
        # except AttributeError:
        #     model_class = model_choices.get_class("espnet")
        model = CTCASRModel(
            vocab_size=vocab_size,
            frontend=frontend,
            encoder=encoder,
        )

        # 8. Initialize

        # generate fake data to init parameters
        speech = jnp.ones([1, 2048], dtype='float')
        speech_lengths = jnp.array([2048])
        text = jnp.ones([1, 128], dtype='int')
        text_lengths = jnp.array([128])
        rng = args.seed  # FIXME(Jiayu): manage seed used in different place!
        rng = jax.random.PRNGKey(rng)
        rng = jax.random.split(rng, 3)
        rng = dict(zip(["skip_layer", "dropout", "params"], rng))
        # TODO: currently hardcode names of required RNG names, might need modifications later
        variables = model.init(rng, speech, speech_lengths, text, text_lengths, False)
        tabular_repr = model.tabulate(rng, speech, speech_lengths, text, text_lengths, False)
        evaluator = model.build_evaluator(token_list)

        return model, variables, tabular_repr, evaluator
