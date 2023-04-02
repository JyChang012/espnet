from typing import Collection, Dict, List, Tuple, Union, Sequence
from math import log2, ceil
from dataclasses import dataclass

import numpy as np
from typeguard import check_argument_types, check_return_type
from jax.tree_util import tree_map


@dataclass
class CommonCollateFn:
    """Functor class of common_collate_fn()"""
    float_pad_value: Union[float, int] = 0.0
    int_pad_value: int = -32768
    not_sequence: Collection[str] = tuple()
    pad_length_to_pow2: bool = True
    pad_batch_to_pow2: bool = True

    def __post_init__(self):
        self.not_sequence = set(self.not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
            self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[Sequence[str], Dict[str, np.ndarray]]:
        return common_collate_fn(data,
                                 float_pad_value=self.float_pad_value,
                                 int_pad_value=self.int_pad_value,
                                 not_sequence=self.not_sequence,
                                 pad_length_to_pow2=self.pad_length_to_pow2,
                                 pad_batch_to_pow2=self.pad_batch_to_pow2)


def pad2pow2(le: int) -> int:
    return 2 ** ceil(log2(le))


def common_collate_fn(
        data: Collection[Tuple[str, Dict[str, np.ndarray]]],
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
        pad_length_to_pow2: bool = False,
        pad_batch_to_pow2: bool = False,
) -> Tuple[Sequence[str], Dict[str, np.ndarray]]:
    """Concatenate ndarray-list to an array and convert to JAX Array.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(array_dict)
        >>> model(**array_dict)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    uttids, data_list = map(tuple, zip(*data))

    keys = tuple(data_list[0])
    keys_set = set(keys)

    assert all(keys_set == set(d) for d in data_list), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in keys
    ), f"*_lengths is reserved: {list(data_list[0])}"

    bsize = len(data)
    n_rows = bsize
    if pad_batch_to_pow2:
        n_rows = pad2pow2(n_rows)

    array_dict = tree_map(lambda *x: x, *data_list)

    for key in keys:
        value_list = array_dict[key]
        if key in not_sequence:
            array_dict[key] = np.array(value_list)
            continue

        if value_list[0].dtype.kind == "i":
            pad_value = int_pad_value
            dtype = np.int32  # JAX use 32 bit int / float by default
        else:
            pad_value = float_pad_value
            dtype = np.float32

        value_list, lengths = map(tuple, zip(*map(lambda arr: (arr.astype(dtype), len(arr)), value_list)))

        maxlen = max(lengths)
        if pad_length_to_pow2:
            maxlen = pad2pow2(maxlen)

        value_arr = np.full((n_rows, maxlen) + tuple(value_list[0].shape[1:]), fill_value=pad_value, dtype=dtype)
        for i, (val, le) in enumerate(zip(value_list, lengths)):
            value_arr[i, :le] = val

        len_arr = np.zeros(n_rows, dtype=np.int32)
        len_arr[:bsize] = lengths

        len_name = f'{key}_lengths'
        array_dict[len_name] = len_arr
        array_dict[key] = value_arr

    out = (uttids, array_dict)
    assert check_return_type(out)
    return out
