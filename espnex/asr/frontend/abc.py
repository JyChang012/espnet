from abc import ABC, abstractmethod
from typing import Tuple

from jax import Array
from flax.linen import Module


class AbsFrontend(Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(
            self,
            input: Array,
            input_lengths: Array
    ) -> Tuple[Array, Array]:
        raise NotImplementedError
