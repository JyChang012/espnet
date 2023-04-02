from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, overload

import flax.linen as nn
import jax
from jax import Array


class AbsEncoder(nn.Module, ABC):
    output_size: int

    @overload  # for type checker
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            prev_states: Array,
            deterministic: bool) -> Tuple[Array, Array, Array]:
        ...

    @overload
    def __call__(
            self,
            xs_pad: Array,
            ilens: Array,
            *,
            deterministic: bool
    ) -> Tuple[Array, Array, None]:
        ...

    @abstractmethod
    def __call__(
        self,
        xs_pad: Array,
        ilens: Array,
        prev_states: Optional[Array] = None,
        deterministic: bool = True
    ) -> Tuple[Array, Array, Optional[Array]]:
        raise NotImplementedError
