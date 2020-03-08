import collections
from functools import reduce
import numpy
import tensorflow


def flatten(iterable):
    return iterable.flatten().tolist() \
        if isinstance(iterable, numpy.ndarray) \
        else (iterable.toArray().tolist()
              if 'Vector' in str(type(iterable))
              else (reduce(lambda left, right: left + flatten(right), iterable, [])
                    if isinstance(iterable, collections.Iterable) and not isinstance(iterable, (dict,) + str)
                    else [iterable]))


def nested_filter(func, iterable):
    return [(nested_filter(func, i)
             if isinstance(i, collections.Iterable) and not isinstance(i, str)
             else i)
            for i in iterable
            if (isinstance(i, collections.Iterable) and (not isinstance(i, str))) or func(i)]


def nested_map(func, iterable):
    ls = list(iterable)
    for i, item in enumerate(iterable):
        ls[i] = \
            nested_map(func, item) \
                if isinstance(item, collections.Iterable) and not isinstance(item, str) \
                else func(item)
    return ls


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
