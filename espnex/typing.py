from typing import TypeVar

from jax import Array

OptionalArray = TypeVar("OptionalArray", Array, None)
