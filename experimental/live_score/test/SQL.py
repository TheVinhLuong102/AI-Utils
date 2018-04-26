from __future__ import print_function

import numpy

from arimo.live_score.SQL import transform

from __init__ import data


data = data(spark_data=False)

print(
    transform(
        data=data.df,
        sql_items=dict(
            int_x='int_x + 1',
            flt_x='flt_x + 10'),
        return_orig_format=True))

print(
    transform(
        data=data.dict,
        sql_items=dict(
            int_x='int_x + 1',
            flt_x='flt_x + 10'),
        return_orig_format=True))

print(
    transform(
        data=data.namespace,
        sql_items=dict(
            int_x='int_x + 2',
            flt_x='flt_x + 20'),
        return_orig_format=False))

print(
    transform(
        data=dict(
            x=1,
            y=None,
            z=numpy.nan),
        sql_items=dict(
            x='x + 2',
            u='COALESCE(y, 20)',
            v='COALESCE(z, 30)',
            w='z'),
        return_orig_format=True))
