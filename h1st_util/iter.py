"""Iterables-related utilities."""


import collections
from typing import Any

import numpy
import tensorflow


__all__ = ('to_iterable',)


def to_iterable(x: Any, iterable_type=tuple) -> collections.Iterable:
    # pylint: disable=invalid-name
    """Return an iterable collection."""
    if isinstance(x, iterable_type):
        return x

    if isinstance(x, collections.Iterable) and \
            (not isinstance(x, (str, tensorflow.Tensor))):
        return iterable_type(x)

    if iterable_type is tuple:
        return (x,)

    if iterable_type is list:
        return [x]

    if iterable_type is set:
        return {x}

    if iterable_type is numpy.ndarray:
        return numpy.array((x,))

    raise TypeError(f'*** INVALID iterable_type {iterable_type} ***')
