from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Generic, TypeVar, Callable, Set, Sequence

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
    ) -> Tuple[Array, Dict[str, Any], float, Any]:  # loss, weight, stats, aux
        raise NotImplementedError

    @abstractmethod
    def collect_feats(
            self,
            *args: Any,
            **kwargs: Any
    ) -> Dict[str, Array]:
        raise NotImplementedError

    # TODO (Jiayu): specify `aux` type using Generic?
    @abstractmethod
    def build_evaluator(
            self,
            *args: Any,
            **kwargs: Any
    ) -> Callable[[float, Dict[str, Any], float, Any], Dict[str, Any]]:
        raise NotImplementedError
