import os

import arimo.backend
from arimo.df.spark import ADF
from arimo.util.date_time import _T_ORD_COL
from arimo.util import fs


SKEWED_DATA_PARQUET_NAME = 'skewed-data.parquet'

SKEWED_DATA_PARQUET_LOCAL_PATH = \
    os.path.join(
        os.path.dirname(os.path.dirname(arimo.backend.__file__)),
        'resources',
        SKEWED_DATA_PARQUET_NAME)

SKEWED_DATA_PARQUET_HDFS_PATH = \
    '/tmp/{}'.format(SKEWED_DATA_PARQUET_NAME)


ID_COL_NAME = 'id'
TIME_COL_NAME = 't'
CONTENT_COL_NAME = 'x'


fs.put(
    from_local=SKEWED_DATA_PARQUET_LOCAL_PATH,
    to_hdfs=SKEWED_DATA_PARQUET_HDFS_PATH,
    is_dir=True, _mv=False)


SPARK_LOCALITY_WAIT = 27   # seconds; setting closer to 0 would result in more even data distrib across executor nodes

if arimo.backend.spark:
    arimo.backend.spark.stop()

if not arimo.backend.chkSpark():
    arimo.backend.init(sparkConf={'spark.locality.wait': SPARK_LOCALITY_WAIT})


adf = ADF.load(
        path=SKEWED_DATA_PARQUET_HDFS_PATH,
        format='parquet',
        verbose=True)(
    ID_COL_NAME,
    'INT({0}) AS {0}'.format(TIME_COL_NAME),
    CONTENT_COL_NAME)


# even distribution across executors
adf.cache()

adf.unpersist()


# skewed distribution across executors
skewed_adf = \
    adf('*',
        'ROW_NUMBER() OVER (PARTITION BY {} ORDER BY {}) AS {}'.format(
            ID_COL_NAME, TIME_COL_NAME, _T_ORD_COL))

skewed_adf.cache()

skewed_adf.unpersist()


# TS ADF has even distribution across executors thanks to chunking
ts_adf = adf('*', iCol=ID_COL_NAME, tCol=TIME_COL_NAME)

ts_adf.cache()

ts_adf.unpersist()
