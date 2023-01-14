from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Generic, TypeVar

import flax
import flax.linen as nn
from flax.linen import Module
from flax.struct import PyTreeNode
from jax import Array


class AbsESPnetModel(Module, ABC):

    @abstractmethod
    def setup(self) -> None:  # must use setup instead of `@compact`
        raise NotImplementedError

    @abstractmethod
    def __call__(  # keyword only args
            self,
            *args: Any,
            **kwargs: Any
    ) -> Tuple[Array, int, Any]:
        raise NotImplementedError

    @abstractmethod
    def collect_feats(
            self,
            *args: Any,
            **kwargs: Any
    ) -> Dict[str, Array]:
        raise NotImplementedError

    # TODO: Add `calculate_stats`, which compute un-jittable metrics
    '''
    @abstractmethod
    def calculate_stats(self, loss: float, weight: int, aux: Any) -> Dict[str, Any]:
        """
        Calculate stats of training output. This method will not be JIT compiled since many metrics like WER are not
        compilable. However, you might call compiled function inside this method.
        """
        raise NotImplementedError
    '''