from typing import Collection, Dict, List, Tuple, Union
from math import log2, ceil

import numpy as np
import torch
from typeguard import check_argument_types, check_return_type

from espnet2.train.collate_fn import CommonCollateFn as TorchCollateFn
from espnet.nets.pytorch_backend.nets_utils import pad_list


class CommonCollateFn(TorchCollateFn):
    """Functor class of common_collate_fn(). Subclass from the torch version."""

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
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
    pad_length_to_pow2: bool = True,
    pad_batch_to_pow2: bool = True,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
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
    data_list = [d for _, d in data]

    assert all(set(data_list[0]) == set(d) for d in data_list), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data_list[0]
    ), f"*_lengths is reserved: {list(data_list[0])}"

    array_dict = {}
    for key in data_list[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        pad_value: Union[int, float]
        if data_list[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data_list]

        # Assume the first axis is length:
        # Batch x (Length, ...)
        bsize = len(array_list)
        if pad_batch_to_pow2:
            bsize = 2 ** ceil(log2(bsize))

        maxlen = max(arr.shape[0] for arr in array_list)

        if pad_length_to_pow2:
            maxlen = 2 ** ceil(log2(maxlen))

        suffix_shape = list(array_list[0].shape[1:])
        dtype = array_list[0].dtype
        array = np.empty([bsize, maxlen] + suffix_shape, dtype=dtype)
        array.fill(pad_value)
        for i, arr in enumerate(array_list):
            le = arr.shape[0]
            array[i, :le] = arr

        array_dict[key] = array

        # lens: (Batch,)
        if key not in not_sequence:
            lens = np.zeros(bsize, dtype=int)
            for i, d in enumerate(data_list):
                lens[i] = d[key].shape[0]
            array_dict[key + "_lengths"] = lens

    out = (uttids, array_dict)
    assert check_return_type(out)
    return out
