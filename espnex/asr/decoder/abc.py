from typing import Tuple
from abc import ABC, abstractmethod

from flax.linen import Module
from jax import Array


class AbsDecoder(Module, ABC):

    @abstractmethod
    def __call__(self,
                 inputs: Array,
                 input_lengths: Array,
                 encoded: Array,
                 encoded_lengths: Array,
                 deterministic: bool,
                 decode: bool = False) -> Tuple[Array, Array]:
        raise NotImplementedError
