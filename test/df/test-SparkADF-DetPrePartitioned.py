from __future__ import print_function

import os
import sys

import arimo.backend
from arimo.df.spark import SparkADF
from arimo.df.spark_from_files import ArrowSparkADF

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import dataset_path, \
    _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY, \
    EQUIPMENT_INSTANCE_ID_COL_NAME, DATE_TIME_COL_NAME


DATE_FROM = '2015-01-01'
DATE_TO = '2015-12-31'
_DATE_FROM___LATE = '2017-01-01'
_DATE_TO___EARLY = '2012-12-31'

_T_ORL_COL_NAME = '_tOrd'


SPARK_DEFAULT_PARALLELISM = 200
SPARK_SQL_SHUFFLE_PARTITIONS = 2001


arimo.backend.initSpark(
    sparkApp='test SparkADF on big ts data',
    sparkConf={'spark.default.parallelism': SPARK_DEFAULT_PARALLELISM,
               'spark.sql.shuffle.partitions': SPARK_SQL_SHUFFLE_PARTITIONS})


arimo.backend.setSpark1Partition1File(on=True)


S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH = \
    dataset_path(
        equipment_general_type_name='disp_case',
        fs='s3', partition='date', format='parquet', compression='snappy')


s3DatePartdParqSnappy__SparkADF = \
    SparkADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        iCol=None, tCol=None,
        detPrePartitioned=False,
        verbose=True)

assert not s3DatePartdParqSnappy__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy__SparkADF.nDetPrePartitions is None

N_DETERMINISTIC_PARTITIONS = s3DatePartdParqSnappy__SparkADF.nPartitions

print(s3DatePartdParqSnappy__SparkADF)
print(s3DatePartdParqSnappy__SparkADF.columns)


s3DatePartdParqSnappy__DetPartd__SparkADF = \
    s3DatePartdParqSnappy__SparkADF(
        '*',
        detPrePartitioned=True)

assert s3DatePartdParqSnappy__DetPartd__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy__DetPartd__SparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__DetPartd__SparkADF.nPartitions == N_DETERMINISTIC_PARTITIONS
print(s3DatePartdParqSnappy__DetPartd__SparkADF)
print(s3DatePartdParqSnappy__DetPartd__SparkADF.columns)


s3DatePartdParqSnappy__DetPartd__TS__SparkADF = \
    s3DatePartdParqSnappy__DetPartd__SparkADF(
        '*',
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME)

assert s3DatePartdParqSnappy__DetPartd__TS__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy__DetPartd__TS__SparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__DetPartd__TS__SparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy__DetPartd__TS__SparkADF)
print(s3DatePartdParqSnappy__DetPartd__TS__SparkADF.columns)


s3DatePartdParqSnappy__TS__SparkADF = \
    s3DatePartdParqSnappy__SparkADF(
        '*',
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME)

assert not s3DatePartdParqSnappy__TS__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy__TS__SparkADF.nDetPrePartitions is None
assert s3DatePartdParqSnappy__TS__SparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy__TS__SparkADF)
print(s3DatePartdParqSnappy__TS__SparkADF.columns)


_s3DatePartdParqSnappy__TS__DetPartd__SparkADF_ = \
    s3DatePartdParqSnappy__TS__SparkADF(
        '*',
        detPrePartitioned=True)

assert _s3DatePartdParqSnappy__TS__DetPartd__SparkADF_.detPrePartitioned
assert _s3DatePartdParqSnappy__TS__DetPartd__SparkADF_.nDetPrePartitions == SPARK_SQL_SHUFFLE_PARTITIONS   # !!!
assert _s3DatePartdParqSnappy__TS__DetPartd__SparkADF_.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(_s3DatePartdParqSnappy__TS__DetPartd__SparkADF_)
print(_s3DatePartdParqSnappy__TS__DetPartd__SparkADF_.columns)


s3DatePartdParqSnappy__DetPartd_TS__SparkADF = \
    s3DatePartdParqSnappy__SparkADF(
        '*',
        detPrePartitioned=True,
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME)

assert s3DatePartdParqSnappy__DetPartd_TS__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy__DetPartd_TS__SparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__DetPartd_TS__SparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy__DetPartd_TS__SparkADF)
print(s3DatePartdParqSnappy__DetPartd_TS__SparkADF.columns)


s3DatePartdParqSnappy_DetPartd__SparkADF = \
    SparkADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        detPrePartitioned=True,
        iCol=None, tCol=None,
        verbose=True)

assert s3DatePartdParqSnappy_DetPartd__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy_DetPartd__SparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_DetPartd__SparkADF.nPartitions == N_DETERMINISTIC_PARTITIONS
print(s3DatePartdParqSnappy_DetPartd__SparkADF)
print(s3DatePartdParqSnappy_DetPartd__SparkADF.columns)


s3DatePartdParqSnappy_DetPartd__TS__SparkADF = \
    s3DatePartdParqSnappy_DetPartd__SparkADF(
        '*',
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME)

assert s3DatePartdParqSnappy_DetPartd__TS__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy_DetPartd__TS__SparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_DetPartd__TS__SparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_DetPartd__TS__SparkADF)
print(s3DatePartdParqSnappy_DetPartd__TS__SparkADF.columns)


s3DatePartdParqSnappy_TS__SparkADF = \
    SparkADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        detPrePartitioned=False,
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME,
        verbose=True)

assert not s3DatePartdParqSnappy_TS__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy_TS__SparkADF.nDetPrePartitions is None
assert s3DatePartdParqSnappy_TS__SparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_TS__SparkADF)
print(s3DatePartdParqSnappy_TS__SparkADF.columns)


_s3DatePartdParqSnappy_TS__DetPartd__SparkADF_ = \
    s3DatePartdParqSnappy_TS__SparkADF(
        '*',
        detPrePartitioned=True)

assert _s3DatePartdParqSnappy_TS__DetPartd__SparkADF_.detPrePartitioned
assert _s3DatePartdParqSnappy_TS__DetPartd__SparkADF_.nDetPrePartitions == SPARK_SQL_SHUFFLE_PARTITIONS   # !!!
assert _s3DatePartdParqSnappy_TS__DetPartd__SparkADF_.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(_s3DatePartdParqSnappy_TS__DetPartd__SparkADF_)
print(_s3DatePartdParqSnappy_TS__DetPartd__SparkADF_.columns)


s3DatePartdParqSnappy_DetPartd_TS__SparkADF = \
    SparkADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        detPrePartitioned=True,
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME,
        verbose=True)

assert s3DatePartdParqSnappy_DetPartd_TS__SparkADF.detPrePartitioned
assert s3DatePartdParqSnappy_DetPartd_TS__SparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_DetPartd_TS__SparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_DetPartd_TS__SparkADF)
print(s3DatePartdParqSnappy_DetPartd_TS__SparkADF.columns)


s3DatePartdParqSnappy__ArrowSparkADF = \
    ArrowSparkADF(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        iCol=None, tCol=None,
        verbose=True)

assert s3DatePartdParqSnappy__ArrowSparkADF.detPrePartitioned
assert s3DatePartdParqSnappy__ArrowSparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__ArrowSparkADF.nPartitions == N_DETERMINISTIC_PARTITIONS
print(s3DatePartdParqSnappy__ArrowSparkADF)
print(s3DatePartdParqSnappy__ArrowSparkADF.columns)


s3DatePartdParqSnappy_TS__ArrowSparkADF = \
    ArrowSparkADF(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=DATE_TIME_COL_NAME,
        verbose=True)

assert s3DatePartdParqSnappy_TS__ArrowSparkADF.detPrePartitioned
assert s3DatePartdParqSnappy_TS__ArrowSparkADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_TS__ArrowSparkADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_TS__ArrowSparkADF)
print(s3DatePartdParqSnappy_TS__ArrowSparkADF.columns)
