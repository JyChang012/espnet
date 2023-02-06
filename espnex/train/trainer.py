# Trainer module
import argparse
import dataclasses
import logging
import time
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any, Callable

import flax.serialization
import humanfriendly
import jax
from flax.core import FrozenDict
from jax import random, Array
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.struct import PyTreeNode
from flax.linen import Module
from optax import GradientTransformation
from typeguard import check_argument_types
from flax.training.train_state import TrainState
from flax.training import checkpoints

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.main_funcs.average_nbest_models import average_nbest_models
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.schedulers.abs_scheduler import (
    AbsBatchStepScheduler,
    AbsEpochStepScheduler,
    AbsScheduler,
    AbsValEpochStepScheduler,
)
from espnet2.torch_utils.add_gradient_noise import add_gradient_noise
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.recursive_op import recursive_average
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnex.train.abs_espnex_model import AbsESPnetModel
from espnet2.train.distributed_utils import DistributedOption
from espnet2.train.reporter import Reporter, SubReporter
from espnet2.utils.build_dataclass import build_dataclass
from espnet2.utils.kwargs2args import kwargs2args

ESPNetTrainState = TrainState  # might need more state


class TrainerOptions(PyTreeNode):
    ngpu: int
    resume: bool
    use_amp: bool
    train_dtype: str
    grad_noise: bool
    accum_grad: int
    grad_clip: float
    grad_clip_type: float
    log_interval: Optional[int]
    no_forward_run: bool
    use_matplotlib: bool
    use_tensorboard: bool
    use_wandb: bool
    output_dir: Union[Path, str]
    max_epoch: int
    seed: int
    sharded_ddp: bool
    patience: Optional[int]
    keep_nbest_models: Union[int, List[int]]
    nbest_averaging_interval: int
    early_stopping_criterion: Sequence[str]
    best_model_criterion: Sequence[Sequence[str]]
    val_scheduler_criterion: Sequence[str]
    unused_parameters: bool
    wandb_model_log_interval: int
    create_graph_in_tensorboard: bool


def train_step(
        state: TrainState,
        batch: Dict[str, Any],
        rng_key: Optional[random.PRNGKeyArray] = None,
):
    if rng_key is not None:
        rng_key_names = 'dropout', 'skip_layer'
        rng_key = random.fold_in(rng_key, state.step)
        rngs = dict(zip(
            rng_key_names,
            random.split(rng_key, len(rng_key_names))
        ))
    else:
        rngs = None

    def loss_fn(params):
        loss, stats, weight, aux = state.apply_fn(
            {'params': params},
            rngs=rngs,
            training=True,
            **batch
        )
        return loss, (stats, weight, aux)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (stats, weight, aux)), grads = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, stats, weight


def pre_eval_step(
        params: FrozenDict,
        batch: Dict[str, Any],
        apply_fn: Callable,
):
    (loss, stats, weight, aux), mod_vars = apply_fn(
        {'params': params},
        training=False,
        mutable='intermediates',
        **batch
    )

    return loss, stats, weight, aux, mod_vars['intermediates']


class Trainer:
    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    def build_options(cls, args: argparse.Namespace) -> TrainerOptions:
        """Build options consumed by train(), eval(), and plot_attention()"""
        assert check_argument_types()
        return build_dataclass(TrainerOptions, args)

    @classmethod
    def run(
            cls,
            state: TrainState,
            train_iter_factory: AbsIterFactory,
            valid_iter_factory: AbsIterFactory,
            prng_key: random.PRNGKeyArray,
            evaluator: Callable[..., Dict[str, Any]],
            plot_attention_iter_factory: Optional[AbsIterFactory],
            trainer_options: TrainerOptions,
            distributed_option: DistributedOption,
    ) -> None:

        if isinstance(trainer_options.keep_nbest_models, int):
            keep_nbest_models = [trainer_options.keep_nbest_models]
        else:
            if len(trainer_options.keep_nbest_models) == 0:
                logging.warning("No keep_nbest_models is given. Change to [1]")
                trainer_options.keep_nbest_models = [1]
            keep_nbest_models = trainer_options.keep_nbest_models

        output_dir = Path(trainer_options.output_dir)
        reporter = Reporter()

        # TODO: support resume training, reported
        start_epoch = reporter.get_epoch() + 1
        if start_epoch == trainer_options.max_epoch + 1:
            logging.warning(
                f"The training has already reached at max_epoch: {start_epoch}"
            )

        if trainer_options.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            train_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "train")
            )
            valid_summary_writer = SummaryWriter(
                str(output_dir / "tensorboard" / "valid")
            )
        else:
            train_summary_writer = valid_summary_writer = None

        jitted_train_step = jax.jit(train_step)
        jitted_eval_step = jax.jit(pre_eval_step, static_argnames='apply_fn')

        start_time = time.perf_counter()
        for iepoch in range(start_epoch, trainer_options.max_epoch + 1):
            if iepoch != start_epoch:
                logging.info(
                    "{}/{} epoch started. Estimated time to finish: {}".format(
                        iepoch,
                        trainer_options.max_epoch,
                        humanfriendly.format_timespan(
                            (time.perf_counter() - start_time)
                            / (iepoch - start_epoch)
                            * (trainer_options.max_epoch - iepoch + 1)
                        ),
                    )
                )
            else:
                logging.info(f"{iepoch}/{trainer_options.max_epoch} epoch started")
            set_all_random_seed(trainer_options.seed + iepoch)

            reporter.set_epoch(iepoch)

            # 1. Train and validation for one-epoch
            with reporter.observe("train") as sub_reporter:
                new_state = cls.train_one_epoch(
                    train_step=jitted_train_step,
                    state=state,
                    iterator=train_iter_factory.build_iter(iepoch),
                    reporter=sub_reporter,
                    summary_writer=train_summary_writer,
                    options=trainer_options,
                    rng_key=prng_key
                )
                state = new_state or state
                all_steps_are_invalid = new_state is None

            with reporter.observe('valid') as sub_reporter:
                cls.validate_one_epoch(
                    jitted_eval_step,
                    state.params,
                    evaluator,
                    state.apply_fn,
                    valid_iter_factory.build_iter(iepoch),
                    sub_reporter,
                    trainer_options
                )
            # TODO: add plot attention

            # 3. Report the results
            # TODO: add tensorboard, wandb, etc
            logging.info(reporter.log_message())
            if train_summary_writer is not None:
                reporter.tensorboard_add_scalar(train_summary_writer, key1="train")
                reporter.tensorboard_add_scalar(valid_summary_writer, key1="valid")
            # 4. Save/Update the checkpoint

            # save checkpoint of last epoch, TODO: save reporter!!!
            checkpoints.save_checkpoint(
                output_dir / 'checkpoints',
                state,
                iepoch,
            )

            # 5. Save and log the model and update the link to the best model
            checkpoints.save_checkpoint(
                output_dir / 'params',
                state.params,
                iepoch,
                'params_'
            )

            # Creates a sym link latest.pth -> {iepoch}epoch.pth
            p = output_dir / "latest"
            if p.is_symlink() or p.exists():
                p.unlink()
            p.symlink_to(f"params/params_{iepoch}")

            _improved = []
            for _phase, k, _mode in trainer_options.best_model_criterion:
                # e.g. _phase, k, _mode = "train", "loss", "min"
                if reporter.has(_phase, k):
                    best_epoch = reporter.get_best_epoch(_phase, k, _mode)
                    # Creates sym links if it's the best result
                    if best_epoch == iepoch:
                        p = output_dir / f"{_phase}.{k}.best"
                        if p.is_symlink() or p.exists():
                            p.unlink()
                        p.symlink_to(f"params/params_{iepoch}")
                        _improved.append(f"{_phase}.{k}")
            if len(_improved) == 0:
                logging.info("There are no improvements in this epoch")
            else:
                logging.info(
                    "The best model has been updated: " + ", ".join(_improved)
                )
            # TODO: wandb

            # 6. Remove the model files excluding n-best epoch and latest epoch
            _removed = []
            # Get the union set of the n-best among multiple criterion
            nbests = set().union(
                *[
                    set(reporter.sort_epochs(ph, k, m)[: max(keep_nbest_models)])
                    for ph, k, m in trainer_options.best_model_criterion
                    if reporter.has(ph, k)
                ]
            )

            # TODO: Generated n-best averaged model

            for e in range(1, iepoch):
                p = output_dir / f"params/params_{e}"
                if p.exists() and e not in nbests:
                    p.unlink()
                    _removed.append(str(p))
            if len(_removed) != 0:
                logging.info("The model files were removed: " + ", ".join(_removed))

            # 7. If any updating haven't happened, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    "The gradients at all steps are invalid in this epoch. "
                    f"Something seems wrong. This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if trainer_options.patience is not None:
                if reporter.check_early_stopping(
                        trainer_options.patience, *trainer_options.early_stopping_criterion
                ):
                    break
        else:
            logging.info(
                f"The training was finished at {trainer_options.max_epoch} epochs "
            )

        # TODO: Generated n-best averaged model

    @classmethod
    def train_one_epoch(
            cls,
            train_step: Callable[[TrainState, Dict[str, Any], random.PRNGKeyArray], Tuple[Array, Array, Any]],
            state: TrainState,
            iterator: Iterable[Tuple[List[str], Dict[str, np.ndarray]]],
            reporter: SubReporter,
            summary_writer,
            options: TrainerOptions,
            rng_key: Optional[random.PRNGKeyArray] = None
    ) -> Optional[TrainState]:
        grad_noise = options.grad_noise
        accum_grad = options.accum_grad
        grad_clip = options.grad_clip
        grad_clip_type = options.grad_clip_type
        log_interval = options.log_interval
        no_forward_run = options.no_forward_run
        ngpu = options.ngpu
        use_wandb = options.use_wandb
        create_graph_in_tensorboard = options.create_graph_in_tensorboard


        log_interval = options.log_interval
        no_forward_run = options.no_forward_run

        if log_interval is None:
            try:
                log_interval = max(len(iterator) // 20, 10)
            except TypeError:
                log_interval = 100
        start_time = time.perf_counter()
        for iiter, (utt_id, batch) in enumerate(
                reporter.measure_iter_time(iterator, "iter_time"), 1
        ):
            with open('itx.txt', 'a') as f:
                f.write(f'{iiter}:\n')
                for k, v in batch.items():
                    if isinstance(v, np.ndarray):
                        f.write(f'Key {k}: array: {v.dtype}[{v.shape}]\n')
                        if v.dtype == np.int64.dtype:
                            f.write(f'\n{v}')
                    else:
                        f.write(f'Key {k}: {v}')
                    f.write('\n')

            # TODO: measure compile / forward / backward time separately
            with reporter.measure_time('step_time'):
                ret = train_step(
                    state,
                    batch,
                    rng_key
                )
            ret = jax.device_get(ret)
            state, stats, weight = ret
            reporter.register(stats, weight)

            reporter.next()
            if iiter % log_interval == 0:
                logging.info(reporter.log_message(-log_interval))
                if summary_writer is not None:
                    reporter.tensorboard_add_scalar(summary_writer, -log_interval)
                if use_wandb:
                    reporter.wandb_log()
        return state


    @classmethod
    def validate_one_epoch(
            cls,
            pre_eval_step: Callable,
            params: FrozenDict,
            evaluator: Callable,
            apply_fn: Callable,
            iterator: Iterable[Dict[str, np.ndarray]],
            reporter: SubReporter,
            options: TrainerOptions,
    ):
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        all_int = []

        for (utt_id, batch) in iterator:
            if no_forward_run:
                continue
            retval = pre_eval_step(params, batch, apply_fn)
            retval = jax.device_get(retval)
            loss, stats, weight, aux, intm = retval

            all_int.append(intm)

            *retval, _ = retval
            stats = evaluator(*retval)

            reporter.register(stats, weight)
            reporter.next()
