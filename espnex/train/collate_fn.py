from typing import Collection, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.train.collate_fn import CommonCollateFn as TorchCollateFn
from espnet.nets.pytorch_backend.nets_utils import pad_list


class CommonCollateFn(TorchCollateFn):
    """Functor class of common_collate_fn(). Subclass from the torch version."""

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, jax.Array]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, jax.Array]]:
    """Concatenate ndarray-list to an array and convert to JAX Array.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        jarray = jnp.array(pad_list(tensor_list, pad_value))
        output[key] = jarray

        # lens: (Batch,)
        if key not in not_sequence:
            lens = jnp.array([d[key].shape[0] for d in data], dtype=int)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    assert check_return_type(output)
    return output
