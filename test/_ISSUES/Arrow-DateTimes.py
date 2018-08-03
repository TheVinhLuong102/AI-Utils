from __future__ import print_function

import datetime
import pandas


PARQUET_PATH = '/tmp/test.parquet'


df0 = pandas.DataFrame(data=dict(date=(datetime.date(2018, 1, 1),)))
print(df0.date[0])


df0.to_parquet(
    fname=PARQUET_PATH,
    engine='pyarrow',
    flavor='spark')


df1 = pandas.read_parquet(
    path=PARQUET_PATH,
    engine='pyarrow')
print(df1.date[0])
