import logging
import sys
from abc import abstractmethod, ABC
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Sequence, Any, Union

import humanfriendly
import numpy as np
from flax.core import FrozenDict
from typeguard import check_argument_types
from torch.utils.data import DataLoader
from jax import Array

from espnet2.train.iterable_dataset import IterableESPnetDataset
from espnet import __version__
from espnet2.samplers.build_batch_sampler import BATCH_TYPES, build_batch_sampler
from espnet2.train.class_choices import ClassChoices
from espnet2.utils import config_argparse
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.train.dataset import DATA_TYPES, AbsDataset, ESPnetDataset
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import (
    humanfriendly_parse_size_or_none,
    int_or_none,
    str2bool,
    str2triple_str,
    str_or_int,
    str_or_none,
)
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump
from espnet.utils.cli_utils import get_commandline_args
from espnex.main_funcs import collect_stats
from espnex.train.abs_espnex_model import AbsESPnetModel


class AbsTask(ABC):
    num_optimizers: int = 1
    trainer = None  # TODO: implement trainer
    class_choices_list: List[ClassChoices] = []

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        pass

    @classmethod
    @abstractmethod
    def build_collate_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, Array]]:
        """Return "collate_fn", which is a callable object and given to DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args, train=True), ...)

        In many cases, you can use our common collate_fn.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
            cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(
            cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as follows,

        >>> from espnet2.train.abs_espnet_model import AbsESPnetModel
        >>> class Model(AbsESPnetModel):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> Tuple[AbsESPnetModel, FrozenDict]:
        raise NotImplementedError

    @classmethod
    def get_parser(cls) -> config_argparse.ArgumentParser:

        class ArgumentDefaultsRawTextHelpFormatter(
            argparse.RawTextHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter,
        ):
            pass

        parser = config_argparse.ArgumentParser(
            ignore_unrecognized=True,  # TODO: temporarily allow unrecognized params for incremental testing
            description="base parser",
            formatter_class=ArgumentDefaultsRawTextHelpFormatter,
        )

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        #  to provide --print_config mode. Instead of it, do as
        parser.set_defaults(required=["output_dir"])

        group = parser.add_argument_group("Common configuration")

        group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )
        group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )
        group.add_argument(
            "--dry_run",
            type=str2bool,
            default=False,
            help="Perform process without training",
        )
        group.add_argument(
            "--iterator_type",
            type=str,
            choices=["sequence", "chunk", "task", "none"],
            default="sequence",
            help="Specify iterator type",
        )

        group.add_argument("--output_dir", type=str_or_none, default=None)
        group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        group.add_argument("--seed", type=int, default=0, help="Random seed")
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs from attention. "
                 "This option makes sense only when attention-based model. "
                 "We can also disable the attention plot by setting it 0",
        )

        group = parser.add_argument_group("distributed training related")
        group.add_argument(
            "--dist_backend",
            default="nccl",
            type=str,
            help="distributed backend",
        )
        group.add_argument(
            "--dist_init_method",
            type=str,
            default="env://",
            help='if init_method="env://", env values of "MASTER_PORT", "MASTER_ADDR", '
                 '"WORLD_SIZE", and "RANK" are referred.',
        )
        group.add_argument(
            "--dist_world_size",
            default=None,
            type=int_or_none,
            help="number of nodes for distributed training",
        )
        group.add_argument(
            "--dist_rank",
            type=int_or_none,
            default=None,
            help="node rank for distributed training",
        )
        group.add_argument(
            # Not starting with "dist_" for compatibility to launch.py
            "--local_rank",
            type=int_or_none,
            default=None,
            help="local rank for distributed training. This option is used if "
                 "--multiprocessing_distributed=false",
        )
        group.add_argument(
            "--dist_master_addr",
            default=None,
            type=str_or_none,
            help="The master address for distributed training. "
                 "This value is used when dist_init_method == 'env://'",
        )
        group.add_argument(
            "--dist_master_port",
            default=None,
            type=int_or_none,
            help="The master port for distributed training"
                 "This value is used when dist_init_method == 'env://'",
        )
        group.add_argument(
            "--dist_launcher",
            default=None,
            type=str_or_none,
            choices=["slurm", "mpi", None],
            help="The launcher type for distributed training",
        )
        group.add_argument(
            "--multiprocessing_distributed",
            default=False,
            type=str2bool,
            help="Use multi-processing distributed training to launch "
                 "N processes per node, which has N GPUs. This is the "
                 "fastest way to use PyTorch for either single node or "
                 "multi node data parallel training",
        )
        group.add_argument(
            "--unused_parameters",
            type=str2bool,
            default=False,
            help="Whether to use the find_unused_parameters in "
                 "torch.nn.parallel.DistributedDataParallel ",
        )
        group.add_argument(
            "--sharded_ddp",
            default=False,
            type=str2bool,
            help="Enable sharded training provided by fairscale",
        )

        group = parser.add_argument_group("collect stats mode related")

        group.add_argument(
            "--collect_stats",
            type=str2bool,
            default=False,
            help='Perform on "collect stats" mode',
        )
        group.add_argument(
            "--write_collected_feats",
            type=str2bool,
            default=False,
            help='Write the output features from the model when "collect stats" mode',
        )

        group = parser.add_argument_group("Trainer related")
        group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
        )
        group.add_argument(
            "--patience",
            type=int_or_none,
            default=None,
            help="Number of epochs to wait without improvement "
                 "before stopping the training",
        )
        group.add_argument(
            "--val_scheduler_criterion",
            type=str,
            nargs=2,
            default=("valid", "loss"),
            help="The criterion used for the value given to the lr scheduler. "
                 'Give a pair referring the phase, "train" or "valid",'
                 'and the criterion name. The mode specifying "min" or "max" can '
                 "be changed by --scheduler_conf",
        )
        group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("valid", "loss", "min"),
            help="The criterion used for judging of early stopping. "
                 'Give a pair referring the phase, "train" or "valid",'
                 'the criterion name and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--best_model_criterion",
            type=str2triple_str,
            nargs="+",
            default=[
                ("train", "loss", "min"),
                ("valid", "loss", "min"),
                ("train", "acc", "max"),
                ("valid", "acc", "max"),
            ],
            help="The criterion used for judging of the best model. "
                 'Give a pair referring the phase, "train" or "valid",'
                 'the criterion name, and the mode, "min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--keep_nbest_models",
            type=int,
            nargs="+",
            default=[10],
            help="Remove previous snapshots excluding the n-best scored epochs",
        )
        group.add_argument(
            "--nbest_averaging_interval",
            type=int,
            default=0,
            help="The epoch interval to apply model averaging and save nbest models",
        )
        group.add_argument(
            "--grad_clip",
            type=float,
            default=5.0,
            help="Gradient norm threshold to clip",
        )
        group.add_argument(
            "--grad_clip_type",
            type=float,
            default=2.0,
            help="The type of the used p-norm for gradient clip. Can be inf",
        )
        group.add_argument(
            "--grad_noise",
            type=str2bool,
            default=False,
            help="The flag to switch to use noise injection to "
                 "gradients during training",
        )
        group.add_argument(
            "--accum_grad",
            type=int,
            default=1,
            help="The number of gradient accumulation",
        )
        group.add_argument(
            "--no_forward_run",
            type=str2bool,
            default=False,
            help="Just only iterating data loading without "
                 "model forwarding and training",
        )
        group.add_argument(
            "--resume",
            type=str2bool,
            default=False,
            help="Enable resuming if checkpoint is existing",
        )
        group.add_argument(
            "--train_dtype",
            default="float32",
            choices=["float16", "float32", "float64"],
            help="Data type for training.",
        )
        group.add_argument(
            "--use_amp",
            type=str2bool,
            default=False,
            help="Enable Automatic Mixed Precision. This feature requires pytorch>=1.6",
        )
        group.add_argument(
            "--log_interval",
            type=int_or_none,
            default=None,
            help="Show the logs every the number iterations in each epochs at the "
                 "training phase. If None is given, it is decided according the number "
                 "of training samples automatically .",
        )
        group.add_argument(
            "--use_matplotlib",
            type=str2bool,
            default=True,
            help="Enable matplotlib logging",
        )
        group.add_argument(
            "--use_tensorboard",
            type=str2bool,
            default=True,
            help="Enable tensorboard logging",
        )
        group.add_argument(
            "--create_graph_in_tensorboard",
            type=str2bool,
            default=False,
            help="Whether to create graph in tensorboard",
        )
        group.add_argument(
            "--use_wandb",
            type=str2bool,
            default=False,
            help="Enable wandb logging",
        )
        group.add_argument(
            "--wandb_project",
            type=str,
            default=None,
            help="Specify wandb project",
        )
        group.add_argument(
            "--wandb_id",
            type=str,
            default=None,
            help="Specify wandb id",
        )
        group.add_argument(
            "--wandb_entity",
            type=str,
            default=None,
            help="Specify wandb entity",
        )
        group.add_argument(
            "--wandb_name",
            type=str,
            default=None,
            help="Specify wandb run name",
        )
        group.add_argument(
            "--wandb_model_log_interval",
            type=int,
            default=-1,
            help="Set the model log period",
        )
        group.add_argument(
            "--detect_anomaly",
            type=str2bool,
            default=False,
            help="Set torch.autograd.set_detect_anomaly",
        )

        group = parser.add_argument_group("Pretraining model related")
        group.add_argument("--pretrain_path", help="This option is obsoleted")
        group.add_argument(
            "--init_param",
            type=str,
            default=[],
            nargs="*",
            help="Specify the file path used for initialization of parameters. "
                 "The format is '<file_path>:<src_key>:<dst_key>:<exclude_keys>', "
                 "where file_path is the model file path, "
                 "src_key specifies the key of model states to be used in the model file, "
                 "dst_key specifies the attribute of the model to be initialized, "
                 "and exclude_keys excludes keys of model states for the initialization."
                 "e.g.\n"
                 "  # Load all parameters"
                 "  --init_param some/where/model.pth\n"
                 "  # Load only decoder parameters"
                 "  --init_param some/where/model.pth:decoder:decoder\n"
                 "  # Load only decoder parameters excluding decoder.embed"
                 "  --init_param some/where/model.pth:decoder:decoder:decoder.embed\n"
                 "  --init_param some/where/model.pth:decoder:decoder:decoder.embed\n",
        )
        group.add_argument(
            "--ignore_init_mismatch",
            type=str2bool,
            default=False,
            help="Ignore size mismatch when loading pre-trained model",
        )
        group.add_argument(
            "--freeze_param",
            type=str,
            default=[],
            nargs="*",
            help="Freeze parameters",
        )

        group = parser.add_argument_group("BatchSampler related")
        group.add_argument(
            "--num_iters_per_epoch",
            type=int_or_none,
            default=None,
            help="Restrict the number of iterations for training per epoch",
        )
        group.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="The mini-batch size used for training. Used if batch_type='unsorted',"
                 " 'sorted', or 'folded'.",
        )
        group.add_argument(
            "--valid_batch_size",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_size is used",
        )
        group.add_argument(
            "--batch_bins",
            type=int,
            default=1000000,
            help="The number of batch bins. Used if batch_type='length' or 'numel'",
        )
        group.add_argument(
            "--valid_batch_bins",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_bins is used",
        )

        group.add_argument("--train_shape_file", type=str, action="append", default=[])
        group.add_argument("--valid_shape_file", type=str, action="append", default=[])


        group = parser.add_argument_group("Sequence iterator related")
        _batch_type_help = ""
        for key, value in BATCH_TYPES.items():
            _batch_type_help += f'"{key}":\n{value}\n'
        group.add_argument(
            "--batch_type",
            type=str,
            default="folded",
            choices=list(BATCH_TYPES),
            help=_batch_type_help,
        )
        group.add_argument(
            "--valid_batch_type",
            type=str_or_none,
            default=None,
            choices=list(BATCH_TYPES) + [None],
            help="If not given, the value of --batch_type is used",
        )
        group.add_argument("--fold_length", type=int, action="append", default=[])
        group.add_argument(
            "--sort_in_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort the samples in each mini-batches by the sample "
                 'lengths. To enable this, "shape_file" must have the length information.',
        )
        group.add_argument(
            "--sort_batch",
            type=str,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort mini-batches by the sample lengths",
        )
        group.add_argument(
            "--multiple_iterator",
            type=str2bool,
            default=False,
            help="Use multiple iterator mode",
        )

        group = parser.add_argument_group("Chunk iterator related")
        group.add_argument(
            "--chunk_length",
            type=str_or_int,
            default=500,
            help="Specify chunk length. e.g. '300', '300,400,500', or '300-400'."
                 "If multiple numbers separated by command are given, "
                 "one of them is selected randomly for each samples. "
                 "If two numbers are given with '-', it indicates the range of the choices. "
                 "Note that if the sequence length is shorter than the all chunk_lengths, "
                 "the sample is discarded. ",
        )
        group.add_argument(
            "--chunk_shift_ratio",
            type=float,
            default=0.5,
            help="Specify the shift width of chunks. If it's less than 1, "
                 "allows the overlapping and if bigger than 1, there are some gaps "
                 "between each chunk.",
        )
        group.add_argument(
            "--num_cache_chunks",
            type=int,
            default=1024,
            help="Shuffle in the specified number of chunks and generate mini-batches "
                 "More larger this value, more randomness can be obtained.",
        )

        group = parser.add_argument_group("Dataset related")
        _data_path_and_name_and_type_help = (
            "Give three words splitted by comma. It's used for the training data. "
            "e.g. '--train_data_path_and_name_and_type some/path/a.scp,foo,sound'. "
            "The first value, some/path/a.scp, indicates the file path, "
            "and the second, foo, is the key name used for the mini-batch data, "
            "and the last, sound, decides the file type. "
            "This option is repeatable, so you can input any number of features "
            "for your task. Supported file types are as follows:\n\n"
        )
        for key, dic in DATA_TYPES.items():
            _data_path_and_name_and_type_help += f'"{key}":\n{dic["help"]}\n\n'

        group.add_argument(
            "--train_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
            help=_data_path_and_name_and_type_help,
        )
        group.add_argument(
            "--valid_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
        )
        group.add_argument(
            "--allow_variable_data_keys",
            type=str2bool,
            default=False,
            help="Allow the arbitrary keys for mini-batch with ignoring "
                 "the task requirements",
        )
        group.add_argument(
            "--max_cache_size",
            type=humanfriendly.parse_size,
            default=0.0,
            help="The maximum cache size for data loader. e.g. 10MB, 20GB.",
        )
        group.add_argument(
            "--max_cache_fd",
            type=int,
            default=32,
            help="The maximum number of file descriptors to be kept "
                 "as opened for ark files. "
                 "This feature is only valid when data type is 'kaldi_ark'.",
        )
        group.add_argument(
            "--valid_max_cache_size",
            type=humanfriendly_parse_size_or_none,
            default=None,
            help="The maximum cache size for validation data loader. e.g. 10MB, 20GB. "
                 "If None, the 5 percent size of --max_cache_size",
        )



        group = parser.add_argument_group("Optimizer related")
        for i in range(1, cls.num_optimizers + 1):
            suf = "" if i == 1 else str(i)

            '''
            group.add_argument(
                f"--optim{suf}",
                type=lambda x: x.lower(),
                default="adadelta",
                choices=list(optim_classes),
                help="The optimizer type",
            )
            '''
            group.add_argument(
                f"--optim{suf}_conf",
                action=NestedDictAction,
                default=dict(),
                help="The keyword arguments for optimizer",
            )
            '''
            group.add_argument(
                f"--scheduler{suf}",
                type=lambda x: str_or_none(x.lower()),
                default=None,
                choices=list(scheduler_classes) + [None],
                help="The lr scheduler type",
            )
            '''
            group.add_argument(
                f"--scheduler{suf}_conf",
                action=NestedDictAction,
                default=dict(),
                help="The keyword arguments for lr scheduler",
            )

        # cls.trainer.add_arguments(parser)
        cls.add_task_arguments(parser)

        # assert check_return_type(parser)
        return parser

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return "required", "print_config", "config", "ngpu"

    @classmethod
    def check_required_command_args(cls, args: argparse.Namespace):
        assert check_argument_types()
        for k in vars(args):
            if "-" in k:
                raise RuntimeError(f'Use "_" instead of "-": parser.get_parser("{k}")')

        required = ", ".join(
            f"--{a}" for a in args.required if getattr(args, a) is None
        )

        if len(required) != 0:
            parser = cls.get_parser()
            parser.print_help(file=sys.stderr)
            p = Path(sys.argv[0]).name
            print(file=sys.stderr)
            print(
                f"{p}: error: the following arguments are required: " f"{required}",
                file=sys.stderr,
            )
            sys.exit(2)

    @classmethod
    def check_task_requirements(
            cls,
            dataset: Union[AbsDataset, IterableESPnetDataset],
            allow_variable_data_keys: bool,
            train: bool,
            inference: bool = False,
    ) -> None:
        """Check if the dataset satisfy the requirement of current Task"""
        assert check_argument_types()
        mes = (
            f"If you intend to use an additional input, modify "
            f'"{cls.__name__}.required_data_names()" or '
            f'"{cls.__name__}.optional_data_names()". '
            f"Otherwise you need to set --allow_variable_data_keys true "
        )

        for k in cls.required_data_names(train, inference):
            if not dataset.has_name(k):
                raise RuntimeError(
                    f'"{cls.required_data_names(train, inference)}" are required for'
                    f' {cls.__name__}. but "{dataset.names()}" are input.\n{mes}'
                )
        if not allow_variable_data_keys:
            task_keys = cls.required_data_names(
                train, inference
            ) + cls.optional_data_names(train, inference)
            for k in dataset.names():
                if k not in task_keys:
                    raise RuntimeError(
                        f"The data-name must be one of {task_keys} "
                        f'for {cls.__name__}: "{k}" is not allowed.\n{mes}'
                    )

    @classmethod
    def build_streaming_iterator(
            cls,
            data_path_and_name_and_type,
            preprocess_fn,
            collate_fn,
            key_file: str = None,
            batch_size: int = 1,
            dtype: str = np.float32,
            num_workers: int = 1,
            allow_variable_data_keys: bool = False,
            ngpu: int = 0,
            inference: bool = False,
    ) -> DataLoader:
        """Build DataLoader using iterable dataset"""
        assert check_argument_types()
        # For backward compatibility for pytorch DataLoader
        if collate_fn is not None:
            kwargs = dict(collate_fn=collate_fn)
        else:
            kwargs = {}

        dataset = IterableESPnetDataset(
            data_path_and_name_and_type,
            float_dtype=dtype,
            preprocess=preprocess_fn,
            key_file=key_file,
        )
        if dataset.apply_utt2category:
            kwargs.update(batch_size=1)
        else:
            kwargs.update(batch_size=batch_size)

        cls.check_task_requirements(
            dataset, allow_variable_data_keys, train=False, inference=inference
        )

        return DataLoader(
            dataset=dataset,
            pin_memory=ngpu > 0,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def main(cls, args: argparse.Namespace = None, cmd: Sequence[str] = None):
        print(get_commandline_args(), file=sys.stderr)
        if args is None:
            parser = cls.get_parser()
            args = parser.parse_args(cmd)
        args.version = __version__

        cls.check_required_command_args(args)
        cls.main_worker(args)

    @classmethod
    def main_worker(cls, args: argparse.Namespace):
        assert check_argument_types()

        model, variables = cls.build_model(args=args)
        assert isinstance(model, AbsESPnetModel), f"model must inherit {AbsESPnetModel.__name__}, but got {type(model)}"

        # TODO: skip build optimizers and schedulers

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w", encoding="utf-8") as f:
            logging.info(
                f'Saving the configuration in {output_dir / "config.yaml"}'
            )
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        if args.dry_run:
            pass
        elif args.collect_stats:
            logging.info(args)
            if args.valid_batch_size is None:
                args.valid_batch_size = args.batch_size

            if len(args.train_shape_file) != 0:
                train_key_file = args.train_shape_file[0]
            else:
                train_key_file = None
            if len(args.valid_shape_file) != 0:
                valid_key_file = args.valid_shape_file[0]
            else:
                valid_key_file = None

            collect_stats(
                model=model,
                variables=variables,
                train_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.train_data_path_and_name_and_type,
                    key_file=train_key_file,
                    batch_size=args.batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                valid_iter=cls.build_streaming_iterator(
                    data_path_and_name_and_type=args.valid_data_path_and_name_and_type,
                    key_file=valid_key_file,
                    batch_size=args.valid_batch_size,
                    dtype=args.train_dtype,
                    num_workers=args.num_workers,
                    allow_variable_data_keys=args.allow_variable_data_keys,
                    ngpu=args.ngpu,
                    preprocess_fn=cls.build_preprocess_fn(args, train=False),
                    collate_fn=cls.build_collate_fn(args, train=False),
                ),
                output_dir=output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
                write_collected_feats=args.write_collected_feats,
            )







