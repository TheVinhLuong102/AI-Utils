from __future__ import print_function

import numpy
import pandas


df0 = pandas.DataFrame(
    data=dict(
        bool=[False, True],
        bool_w_nan=[False, True],
        bool_w_none=[False, True],
        flt=[-3., 3.],
        flt_w_nan=[-3., 3.],
        flt_w_none=[-3., 3.],
        int=[-1, 1],
        int_w_nan=[-1, 1],
        int_w_none=[-1, 1],
        str=['z', 'b'],
        str_w_nan=['z', 'b'],
        str_w_none=['z', 'b']))

print('\n', df0)

df0.info(
    verbose=True,
    max_cols=None,
    memory_usage=True,
    null_counts=True)


df0.loc[2] = dict(
    bool=False,
    bool_w_nan=numpy.nan,
    bool_w_none=None,
    flt=0.,
    flt_w_nan=numpy.nan,
    flt_w_none=None,
    int=0,
    int_w_nan=numpy.nan,
    int_w_none=None,
    str='a',
    str_w_nan=numpy.nan,
    str_w_none=None)

print('\n', df0)

df0.info(
    verbose=True,
    max_cols=None,
    memory_usage=True,
    null_counts=True)


df1 = pandas.DataFrame(
    data=dict(
        bool=[False, False, True],
        bool_w_nan=[False, numpy.nan, True],
        bool_w_none=[False, None, True],
        flt=[-3., 0., 3.],
        flt_w_nan=[-3., numpy.nan, 3.],
        flt_w_none=[-3., None, 3.],
        int=[-1, 0, 1],
        int_w_nan=[-1, numpy.nan, 1],
        int_w_none=[-1, None, 1],
        str=['z', 'a', 'b'],
        str_w_nan=['z', numpy.nan, 'b'],
        str_w_none=['z', None, 'b']))

print('\n', df1)

df1.info(
    verbose=True,
    max_cols=None,
    memory_usage=True,
    null_counts=True)


df1.drop(
    labels=[1],
    axis='index',
    level=None,
    inplace=True,
    errors='raise')

print('\n', df1)

df1.info(
    verbose=True,
    max_cols=None,
    memory_usage=True,
    null_counts=True)
