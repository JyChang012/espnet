from abc import ABC, abstractmethod
from typing import Optional, Tuple

import flax.linen as nn
import jax


class AbsEncoder(nn.Module, ABC):
    @abstractmethod
    def output_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        xs_pad: jax.Array,
        ilens: jax.Array,
        prev_states: jax.Array = None,
    ) -> Tuple[jax.Array, jax.Array, Optional[jax.Array]]:
        raise NotImplementedError
