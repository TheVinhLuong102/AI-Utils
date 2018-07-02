from __future__ import print_function

import datetime
import itertools
import os
import pandas

import sys
if sys.version_info.major == 3:
    from functools import reduce

import arimo.backend
from arimo.df.spark import SparkADF
from arimo.df.from_files import ArrowADF


N_IDS = 3
N_TIME_STEPS_PER_ID = 20
N_ROWS = N_IDS * N_TIME_STEPS_PER_ID

ID_RANGE = range(1, N_IDS + 1)
TIME_ORD_RANGE = range(1, N_TIME_STEPS_PER_ID + 1)

YEAR = datetime.date.today().year

ID_COL_NAME = 'i'
TIME_COL_NAME = 't'
CONTENT_COL_NAME = 'x'

SPLIT_ADF_T_CHUNK_LEN = 3
ORIG_T_CHUNK_LEN = 2 * SPLIT_ADF_T_CHUNK_LEN


PARQUET_LOCAL_PATH = \
    os.path.join(
        os.path.dirname(os.path.dirname(arimo.backend.__file__)),
        'resources',
        'skewed-data.parquet')


df = pandas.DataFrame(
    data={
        ID_COL_NAME:
            reduce(
                lambda x, y: x + y,
                [N_TIME_STEPS_PER_ID * ['i{}'.format(i)]
                 for i in ID_RANGE]),

        TIME_COL_NAME:
            [datetime.date(year=YEAR, month=m, day=d)
             for m, d in itertools.product(ID_RANGE, TIME_ORD_RANGE)],

        CONTENT_COL_NAME:
            range(11, N_ROWS + 11)
    }).sample(
        n=N_ROWS,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None)


spark_adf = \
    SparkADF.create(
        data=df,
        iCol=ID_COL_NAME,
        tCol=TIME_COL_NAME,
        tChunkLen=ORIG_T_CHUNK_LEN)


spark_adf_1, spark_adf_2 = spark_adf.split(.68, .32)

spark_adf_1.tChunkLen = spark_adf_2.tChunkLen = SPLIT_ADF_T_CHUNK_LEN

print('Split SparkADF #1:')
spark_adf_1.show(N_ROWS)

print('Split SparkADF #2:')
spark_adf_2.show(N_ROWS)


arrow_adf = ArrowADF(path=PARQUET_LOCAL_PATH)
print(arrow_adf)


arrow_adf_1, arrow_adf_2 = arrow_adf.split(.68, .32)

print('Split ArrowADF #1: {}'.format(arrow_adf_1))
print('Split ArrowADF #2: {}'.format(arrow_adf_2))
