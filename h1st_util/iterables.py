import collections
from functools import reduce
import numpy
import tensorflow


def to_iterable(x, iterable_type=tuple):
    if isinstance(x, iterable_type):
        return x
    elif isinstance(x, collections.Iterable) and (not isinstance(x, (str, tensorflow.Tensor))):
        return iterable_type(x)
    elif iterable_type is tuple:
        return x,
    elif iterable_type is list:
        return [x]
    elif iterable_type is set:
        return {x}
    elif iterable_type is numpy.ndarray:
        return numpy.array((x,))
