# Trainer module
import argparse
import dataclasses
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import is_dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any, Callable

import flax.serialization
import humanfriendly
import jax
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict
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
from espnex.main_funcs.calculate_all_attentions import calculate_all_attentions
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
    # grad_clip: float  # moved to GradTransformation
    # grad_clip_type: float
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
        rng_key_names: Optional[Sequence[str]] = None,
):
    if rng_key is not None:
        assert rng_key_names is not None, 'Rng key is not None but no rng key names are provided!'
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
    loss, stats, weight, aux = apply_fn(
        {'params': params},
        **batch,
        training=False
    )

    return loss, stats, weight, aux


def get_attention_weight_step(
        params: FrozenDict,
        sample: Dict[str, Any],
        apply_fn: Callable
):
    _, new_vars = apply_fn(dict(params=params), **sample, training=False, mutable='intermediates')
    intermediates = new_vars['intermediates']
    weights = flatten_dict(intermediates)
    weights = {'.'.join(k): v[0] if isinstance(v, Sequence) else v
               for k, v in weights.items()
               if 'attention_weight' in k[-1]}
    return weights


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
            rng_key_names: Sequence[str],
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

        jitted_train_step = jax.jit(partial(train_step, rng_key_names=rng_key_names))
        jitted_eval_step = jax.jit(partial(pre_eval_step, apply_fn=state.apply_fn))
        jitted_get_attention_weight_step = jax.jit(
            partial(get_attention_weight_step, apply_fn=state.apply_fn)
        )

        # TODO: return decoded for debugging
        evaluator = partial(evaluator, return_decoded=True)

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
                results = cls.validate_one_epoch(
                    jitted_eval_step,
                    state.params,
                    evaluator,
                    valid_iter_factory.build_iter(iepoch),
                    sub_reporter,
                    trainer_options
                )

                p = output_dir / 'val_decoded' / f'ep{iepoch}'
                p.parent.mkdir(parents=True, exist_ok=True)
                to_write = zip(*results.values())
                to_write = map(lambda x: '\n'.join(x), to_write)
                to_write = '\n\n'.join(to_write)
                with open(p, 'w') as f:
                    f.write(f'Header: {" ".join(results.keys())}\n\n\n')
                    f.write(to_write)

            if plot_attention_iter_factory is not None:
                with reporter.observe("att_plot") as sub_reporter:
                    cls.plot_attention(
                        state.params,
                        get_attention_weight_step=jitted_get_attention_weight_step,
                        output_dir=output_dir / "att_ws",
                        summary_writer=train_summary_writer,
                        iterator=plot_attention_iter_factory.build_iter(iepoch),
                        reporter=sub_reporter,
                        options=trainer_options,
                    )

            # 3. Report the results
            # TODO: add wandb, etc
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
            '''
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
            '''

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
            iterator: Iterable[Dict[str, np.ndarray]],
            reporter: SubReporter,
            options: TrainerOptions,
    ):
        ngpu = options.ngpu
        no_forward_run = options.no_forward_run

        results = defaultdict(list)
        for (utt_id, batch) in iterator:
            if no_forward_run:
                continue
            retval = pre_eval_step(params, batch)
            retval = jax.device_get(retval)
            loss, stats, weight, aux = retval  # TODO: currently not using intermediates!

            # *retval, _ = retval
            stats, aux = evaluator(*retval)

            results['indices'].extend(utt_id)
            for k, v in aux.items():
                results[k].extend(v)

            reporter.register(stats, weight)
            reporter.next()

        return results

    @classmethod
    def plot_attention(
            cls,
            params: FrozenDict,
            get_attention_weight_step: Callable[[FrozenDict, Dict[str, Any]], Dict[str, Any]],
            output_dir: Optional[Path],
            summary_writer,
            iterator: Iterable[Tuple[List[str], Dict[str, np.ndarray]]],
            reporter: SubReporter,
            options: TrainerOptions,
    ) -> None:
        import matplotlib

        no_forward_run = options.no_forward_run

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )

            # batch["utt_id"] = ids
            if no_forward_run:
                continue

            att_dict = calculate_all_attentions(
                get_attention_weight_step,
                params,
                batch
            )

            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):
                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim == 4:
                        # In multispkr_asr model case, the dimension could be 4.
                        att_w = np.concatenate(
                            [att_w[i] for i in range(att_w.shape[0])], axis=0
                        )
                    elif att_w.ndim > 4 or att_w.ndim == 1:
                        raise RuntimeError(f"Must be 2, 3 or 4 dimension: {att_w.ndim}")

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    if output_dir is not None:
                        p = output_dir / id_ / f"{k}.{reporter.get_epoch()}ep.png"
                        p.parent.mkdir(parents=True, exist_ok=True)
                        fig.savefig(p)

                    if summary_writer is not None:
                        summary_writer.add_figure(
                            f"{k}_{id_}", fig, reporter.get_epoch()
                        )

                    if options.use_wandb:
                        import wandb

                        wandb.log({f"attention plot/{k}_{id_}": wandb.Image(fig)})
            reporter.next()





