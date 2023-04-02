from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from flax.linen import Module
from jax import Array


class AbsFrontend(Module, ABC):
    deterministic: Optional[bool] = None

    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self, input: Array, input_lengths: Array, deterministic: Optional[bool] = None
    ) -> Tuple[Array, Array]:
        raise NotImplementedError
