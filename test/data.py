from __future__ import print_function

import os

from arimo.util import Namespace
from arimo.util.iterables import to_iterable
from arimo.util.aws import key_pair, s3


_AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY = key_pair('PanaAP-CC')


REFRIG_EQ_GEN_TYPE_NAME = 'refrig'
DISP_CASE_EQ_GEN_TYPE_NAME = 'disp_case'


DATASET_NAMES = \
    Namespace(**{
        REFRIG_EQ_GEN_TYPE_NAME: 'REFRIG---inverter_4_multi_comp_refrigerator',
        DISP_CASE_EQ_GEN_TYPE_NAME: 'DISP_CASE---ex_display_case'})


TEST_RESOURCES_S3_DIR_PATH = 's3://arimo-panasonic-ap/data/test-resources'

_TEST_RESOURCES_S3A_AUTH_DIR_PATH = \
    s3.s3a_path_with_auth(
        s3_path=TEST_RESOURCES_S3_DIR_PATH,
        access_key_id=_AWS_ACCESS_KEY_ID,
        secret_access_key=_AWS_SECRET_ACCESS_KEY)


_TMP_DIR_PATH = '/tmp'


def dataset_path(
        equipment_general_type_name='disp_case',
        fs='s3', partition='date', format='parquet', compression='snappy'):
    assert format in ('parquet', 'orc')

    if compression is None:
        compression = 'uncompressed'
    else:
        compression = compression.lower()
        if compression == 'none':
            compression = 'uncompressed'

    _sub_path = \
        '{}.{}-partitioned.{}.{}'.format(
            DATASET_NAMES[equipment_general_type_name],
            'rand' if partition is None \
                   else '-'.join(s.upper() for s in to_iterable(partition)),
            format,
            compression)

    if fs == 'local':
        return os.path.join(_TMP_DIR_PATH, _sub_path)

    elif fs == 'hdfs':
        return os.path.join('hdfs:' + _TMP_DIR_PATH, _sub_path)

    elif fs == 's3':
        return os.path.join(TEST_RESOURCES_S3_DIR_PATH, _sub_path)

    else:
        assert fs == 's3a'
        return os.path.join(_TEST_RESOURCES_S3A_AUTH_DIR_PATH, _sub_path)


SMALL_DATA_LOCAL_PATH = dataset_path(equipment_general_type_name=REFRIG_EQ_GEN_TYPE_NAME, fs='local')
SMALL_DATA_HDFS_PATH = dataset_path(equipment_general_type_name=REFRIG_EQ_GEN_TYPE_NAME, fs='hdfs')
SMALL_DATA_S3_PATH = dataset_path(equipment_general_type_name=REFRIG_EQ_GEN_TYPE_NAME, fs='s3')

BIG_DATA_LOCAL_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='local')
BIG_DATA_HDFS_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='hdfs')
BIG_DATA_S3_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3')


BIG_DATA_S3_PARQUET_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3', format='parquet')
BIG_DATA_S3A_AUTH_PARQUET_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3a', format='parquet')

BIG_DATA_S3_RAND_PARTITIONED_PARQUET_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3', partition='rand', format='parquet')
BIG_DATA_S3A_AUTH_RAND_PARTITIONED_PARQUET_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3a', partition='rand', format='parquet')

BIG_DATA_HDFS_PARQUET_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='hdfs', format='parquet')
BIG_DATA_RAND_PARTITIONED_HDFS_PARQUET_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='hdfs', partition='rand', format='parquet')


BIG_DATA_S3_ORC_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3', format='orc')
BIG_DATA_S3A_AUTH_ORC_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3a', format='orc')

BIG_DATA_S3_RAND_PARTITIONED_ORC_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3', partition='rand', format='orc')
BIG_DATA_S3A_AUTH_RAND_PARTITIONED_ORC_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='s3a', partition='rand', format='orc')

BIG_DATA_HDFS_ORC_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='hdfs', format='orc')
BIG_DATA_RAND_PARTITIONED_HDFS_ORC_PATH = dataset_path(equipment_general_type_name=DISP_CASE_EQ_GEN_TYPE_NAME, fs='hdfs', partition='rand', format='orc')
