from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, overload

import flax.linen as nn
import jax
from jax import Array


class AbsEncoder(nn.Module, ABC):
    deterministic: Optional[bool] = None

    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @overload  # for type checker
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            prev_states: Array,
            deterministic: Optional[bool]) -> Tuple[Array, Array, Array]:
        ...

    @overload
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            *,
            deterministic: Optional[bool]
    ) -> Tuple[Array, Array, None]:
        ...

    @abstractmethod
    def __call__(
        self,
        xs_pad: Array,
        ilens: Array,
        prev_states: Optional[Array] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Array, Array, Optional[Array]]:
        raise NotImplementedError
