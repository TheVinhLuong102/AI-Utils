from __future__ import print_function

import arimo.backend
from arimo.df.spark import ADF
from arimo.df.spark_on_files import FileADF
import arimo.debug

from arimo.IoT.DataAdmin import Project as IoTDataAdminProject

from PanasonicColdChain import _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY
from data import dataset_path


arimo.debug.ON = True


DATE_FROM = '2015-01-01'
DATE_TO = '2015-12-31'
_DATE_FROM___LATE = '2017-01-01'
_DATE_TO___EARLY = '2012-12-31'

_T_ORL_COL_NAME = '_tOrd'


SPARK_DEFAULT_PARALLELISM = 200
SPARK_SQL_SHUFFLE_PARTITIONS = 2001

arimo.backend.init(
    sparkApp='test ADF on big ts data',
    sparkConf={'spark.default.parallelism': SPARK_DEFAULT_PARALLELISM,
               'spark.sql.shuffle.partitions': SPARK_SQL_SHUFFLE_PARTITIONS})


S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH = \
    dataset_path(
        equipment_general_type_name='disp_case',
        fs='s3', partition='date', format='parquet', compression='snappy')


s3DatePartdParqSnappy__ADF = \
    ADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        iCol=None, tCol=None,
        detPrePartitioned=False,
        verbose=True)

assert not s3DatePartdParqSnappy__ADF.detPrePartitioned
assert s3DatePartdParqSnappy__ADF.nDetPrePartitions is None

N_DETERMINISTIC_PARTITIONS = s3DatePartdParqSnappy__ADF.nPartitions

print(s3DatePartdParqSnappy__ADF)
print(s3DatePartdParqSnappy__ADF.columns)


s3DatePartdParqSnappy__DetPartd__ADF = \
    s3DatePartdParqSnappy__ADF(
        '*',
        detPrePartitioned=True)

assert s3DatePartdParqSnappy__DetPartd__ADF.detPrePartitioned
assert s3DatePartdParqSnappy__DetPartd__ADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__DetPartd__ADF.nPartitions == N_DETERMINISTIC_PARTITIONS
print(s3DatePartdParqSnappy__DetPartd__ADF)
print(s3DatePartdParqSnappy__DetPartd__ADF.columns)


s3DatePartdParqSnappy__DetPartd__TS__ADF = \
    s3DatePartdParqSnappy__DetPartd__ADF(
        '*',
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME)

assert s3DatePartdParqSnappy__DetPartd__TS__ADF.detPrePartitioned
assert s3DatePartdParqSnappy__DetPartd__TS__ADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__DetPartd__TS__ADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy__DetPartd__TS__ADF)
print(s3DatePartdParqSnappy__DetPartd__TS__ADF.columns)


s3DatePartdParqSnappy__TS__ADF = \
    s3DatePartdParqSnappy__ADF(
        '*',
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME)

assert not s3DatePartdParqSnappy__TS__ADF.detPrePartitioned
assert s3DatePartdParqSnappy__TS__ADF.nDetPrePartitions is None
assert s3DatePartdParqSnappy__TS__ADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy__TS__ADF)
print(s3DatePartdParqSnappy__TS__ADF.columns)


_s3DatePartdParqSnappy__TS__DetPartd__ADF_ = \
    s3DatePartdParqSnappy__TS__ADF(
        '*',
        detPrePartitioned=True)

assert _s3DatePartdParqSnappy__TS__DetPartd__ADF_.detPrePartitioned
assert _s3DatePartdParqSnappy__TS__DetPartd__ADF_.nDetPrePartitions == SPARK_SQL_SHUFFLE_PARTITIONS   # !!!
assert _s3DatePartdParqSnappy__TS__DetPartd__ADF_.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(_s3DatePartdParqSnappy__TS__DetPartd__ADF_)
print(_s3DatePartdParqSnappy__TS__DetPartd__ADF_.columns)


s3DatePartdParqSnappy__DetPartd_TS__ADF = \
    s3DatePartdParqSnappy__ADF(
        '*',
        detPrePartitioned=True,
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME)

assert s3DatePartdParqSnappy__DetPartd_TS__ADF.detPrePartitioned
assert s3DatePartdParqSnappy__DetPartd_TS__ADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__DetPartd_TS__ADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy__DetPartd_TS__ADF)
print(s3DatePartdParqSnappy__DetPartd_TS__ADF.columns)


s3DatePartdParqSnappy_DetPartd__ADF = \
    ADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        detPrePartitioned=True,
        iCol=None, tCol=None,
        verbose=True)

assert s3DatePartdParqSnappy_DetPartd__ADF.detPrePartitioned
assert s3DatePartdParqSnappy_DetPartd__ADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_DetPartd__ADF.nPartitions == N_DETERMINISTIC_PARTITIONS
print(s3DatePartdParqSnappy_DetPartd__ADF)
print(s3DatePartdParqSnappy_DetPartd__ADF.columns)


s3DatePartdParqSnappy_DetPartd__TS__ADF = \
    s3DatePartdParqSnappy_DetPartd__ADF(
        '*',
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME)

assert s3DatePartdParqSnappy_DetPartd__TS__ADF.detPrePartitioned
assert s3DatePartdParqSnappy_DetPartd__TS__ADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_DetPartd__TS__ADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_DetPartd__TS__ADF)
print(s3DatePartdParqSnappy_DetPartd__TS__ADF.columns)


s3DatePartdParqSnappy_TS__ADF = \
    ADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        detPrePartitioned=False,
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME,
        verbose=True)

assert not s3DatePartdParqSnappy_TS__ADF.detPrePartitioned
assert s3DatePartdParqSnappy_TS__ADF.nDetPrePartitions is None
assert s3DatePartdParqSnappy_TS__ADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_TS__ADF)
print(s3DatePartdParqSnappy_TS__ADF.columns)


_s3DatePartdParqSnappy_TS__DetPartd__ADF_ = \
    s3DatePartdParqSnappy_TS__ADF(
        '*',
        detPrePartitioned=True)

assert _s3DatePartdParqSnappy_TS__DetPartd__ADF_.detPrePartitioned
assert _s3DatePartdParqSnappy_TS__DetPartd__ADF_.nDetPrePartitions == SPARK_SQL_SHUFFLE_PARTITIONS   # !!!
assert _s3DatePartdParqSnappy_TS__DetPartd__ADF_.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(_s3DatePartdParqSnappy_TS__DetPartd__ADF_)
print(_s3DatePartdParqSnappy_TS__DetPartd__ADF_.columns)


s3DatePartdParqSnappy_DetPartd_TS__ADF = \
    ADF.load(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        format='parquet',
        detPrePartitioned=True,
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME,
        verbose=True)

assert s3DatePartdParqSnappy_DetPartd_TS__ADF.detPrePartitioned
assert s3DatePartdParqSnappy_DetPartd_TS__ADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_DetPartd_TS__ADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_DetPartd_TS__ADF)
print(s3DatePartdParqSnappy_DetPartd_TS__ADF.columns)


s3DatePartdParqSnappy__FilesADF = \
    FileADF(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        iCol=None, tCol=None,
        verbose=True)

assert s3DatePartdParqSnappy__FilesADF.detPrePartitioned
assert s3DatePartdParqSnappy__FilesADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy__FilesADF.nPartitions == N_DETERMINISTIC_PARTITIONS
print(s3DatePartdParqSnappy__FilesADF)
print(s3DatePartdParqSnappy__FilesADF.columns)


s3DatePartdParqSnappy_TS__FilesADF = \
    FileADF(
        path=S3_DATE_PARTITIONED_PARQUET_SNAPPY_DATA_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        iCol=IoTDataAdminProject._EQUIPMENT_INSTANCE_ID_COL_NAME, tCol=IoTDataAdminProject._DATE_TIME_COL_NAME,
        verbose=True)

assert s3DatePartdParqSnappy_TS__FilesADF.detPrePartitioned
assert s3DatePartdParqSnappy_TS__FilesADF.nDetPrePartitions == N_DETERMINISTIC_PARTITIONS
assert s3DatePartdParqSnappy_TS__FilesADF.nPartitions == SPARK_SQL_SHUFFLE_PARTITIONS
print(s3DatePartdParqSnappy_TS__FilesADF)
print(s3DatePartdParqSnappy_TS__FilesADF.columns)
