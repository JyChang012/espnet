from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

from jax import Array


class InversibleInterface(ABC):

    @abstractmethod
    def inverse(
            self,
            input: Array,
            input_lengths: Optional[Array] = None
    ) -> Tuple[Array, Optional[Array]]:
        # return output, output_lengths
        raise NotImplementedError
