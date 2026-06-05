"""Typing aliases and protocols for qsimov."""

from typing import Sequence, Union, TypeVar, Callable, Protocol

T = TypeVar("T")

arrayNd = Union[Sequence[T], Sequence["arrayNd[T]"]]
"""A type alias for a nested array of arbitrary depth."""

array1d = arrayNd[T]
"""A type alias for a 1D array."""

array2d = arrayNd[array1d[T]]
"""A type alias for a 2D array."""

array3d = arrayNd[array2d[T]]
"""A type alias for a 3D array."""

array4d = arrayNd[array3d[T]]
"""A type alias for a 4D array."""


class namedcallable(Protocol):
    """A callable with a name attribute (implements __name__ and __call__, for
    example, a python function)."""

    __name__: str
    __call__: Callable
