from __future__ import division, print_function

import os

from arimo.df.from_files import ArrowADF
from arimo.df.spark import SparkADF
from arimo.df.spark_from_files import ArrowSparkADF
from arimo.util import fs
from arimo.util.aws import s3
import arimo.debug

from arimo.IoT.DataAdmin import project

from data import \
    SMALL_DATA_LOCAL_PATH, SMALL_DATA_HDFS_PATH, SMALL_DATA_S3_PATH, \
    BIG_DATA_LOCAL_PATH, BIG_DATA_HDFS_PATH, BIG_DATA_S3_PATH

arimo.debug.ON = True


P = project('PanaAP-CC')

_AWS_ACCESS_KEY_ID = P.params.s3.access_key_id
_AWS_SECRET_ACCESS_KEY = P.params.s3.secret_access_key


EQUIPMENT_INSTANCE_ID_COL_NAME = 'equipment_instance_id'
DATE_TIME_COL_NAME = 'date_time'


if not os.path.isdir(SMALL_DATA_LOCAL_PATH):
    s3.sync(
        from_dir_path=SMALL_DATA_S3_PATH,
        to_dir_path=SMALL_DATA_LOCAL_PATH,
        delete=True, quiet=False,
        access_key_id=_AWS_ACCESS_KEY_ID, secret_access_key=_AWS_SECRET_ACCESS_KEY,
        verbose=True)


smallLocalArrowADF = ArrowADF(
    path=SMALL_DATA_LOCAL_PATH,
    iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
    tCol=DATE_TIME_COL_NAME)
print(smallLocalArrowADF)

print(smallLocalArrowADF.approxNRows)
print(smallLocalArrowADF)

print(smallLocalArrowADF.nRows)
print(smallLocalArrowADF)

print(smallLocalArrowADF.reprSample.shape)


smallS3ArrowADF = ArrowADF(
    path=SMALL_DATA_S3_PATH,
    aws_access_key_id=_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
    iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
    tCol=DATE_TIME_COL_NAME)
print(smallS3ArrowADF)

print(smallS3ArrowADF.approxNRows)
print(smallS3ArrowADF)

# print(smallS3ArrowADF.nRows)   # SLOW
# print(smallS3ArrowADF)

print(smallS3ArrowADF.reprSample.shape)


if not os.path.isdir(BIG_DATA_LOCAL_PATH):
    s3.sync(
        from_dir_path=BIG_DATA_S3_PATH,
        to_dir_path=BIG_DATA_LOCAL_PATH,
        delete=True, quiet=False,
        access_key_id=_AWS_ACCESS_KEY_ID, secret_access_key=_AWS_SECRET_ACCESS_KEY,
        verbose=True)


bigLocalArrowADF = ArrowADF(
    path=BIG_DATA_LOCAL_PATH,
    iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
    tCol=DATE_TIME_COL_NAME)
print(bigLocalArrowADF)

print(bigLocalArrowADF.approxNRows)
print(bigLocalArrowADF)

print(bigLocalArrowADF.nRows)
print(bigLocalArrowADF)

print(bigLocalArrowADF.reprSample.shape)


bigS3ArrowADF = ArrowADF(
    path=BIG_DATA_S3_PATH,
    aws_access_key_id=_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
    iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
    tCol=DATE_TIME_COL_NAME)
print(bigS3ArrowADF)

print(bigS3ArrowADF.approxNRows)
print(bigS3ArrowADF)

# print(bigS3ArrowADF.nRows)   # SLOW
# print(bigS3ArrowADF)

print(bigS3ArrowADF.reprSample.shape)


if fs._ON_LINUX_CLUSTER_WITH_HDFS:
    if not fs.hdfs_client.test(
            path=SMALL_DATA_LOCAL_PATH,
            exists=True,
            directory=True):
        fs.put(
            from_local=SMALL_DATA_LOCAL_PATH,
            to_hdfs=SMALL_DATA_LOCAL_PATH,
            is_dir=True, _mv=False)


    smallHdfsArrowADF = ArrowADF(
        path=SMALL_DATA_HDFS_PATH,
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
        tCol=DATE_TIME_COL_NAME)
    print(smallHdfsArrowADF)

    print(smallHdfsArrowADF.approxNRows)
    print(smallHdfsArrowADF)

    print(smallHdfsArrowADF.nRows)
    print(smallHdfsArrowADF)

    print(smallHdfsArrowADF.reprSample.shape)


    smallHdfsSparkADF = \
        SparkADF.load(
            path=SMALL_DATA_HDFS_PATH,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(smallHdfsSparkADF)

    print(smallHdfsSparkADF.nRows)
    print(smallHdfsSparkADF)

    print(smallHdfsSparkADF.reprSample.shape)


    smallHdfsArrowSparkADF = \
        ArrowSparkADF(
            path=SMALL_DATA_HDFS_PATH,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(smallHdfsArrowSparkADF)

    print(smallHdfsArrowSparkADF.nRows)
    print(smallHdfsArrowSparkADF)

    print(smallHdfsArrowSparkADF.reprSample.shape)


    smallS3SparkADF = \
        SparkADF.load(
            path=SMALL_DATA_S3_PATH,
            aws_access_key_id=_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(smallS3SparkADF)

    print(smallS3SparkADF.nRows)
    print(smallS3SparkADF)

    print(smallS3SparkADF.reprSample.shape)


    smallS3ArrowSparkADF = \
        ArrowSparkADF(
            path=SMALL_DATA_S3_PATH,
            aws_access_key_id=_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(smallS3ArrowSparkADF)

    print(smallS3ArrowSparkADF.nRows)
    print(smallS3ArrowSparkADF)

    print(smallS3ArrowSparkADF.reprSample.shape)


    if not fs.hdfs_client.test(
            path=BIG_DATA_LOCAL_PATH,
            exists=True,
            directory=True):
        fs.put(
            from_local=BIG_DATA_LOCAL_PATH,
            to_hdfs=BIG_DATA_LOCAL_PATH,
            is_dir=True, _mv=False)


    bigHdfsArrowADF = ArrowADF(
        path=BIG_DATA_HDFS_PATH,
        iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
        tCol=DATE_TIME_COL_NAME)
    print(bigHdfsArrowADF)

    print(bigHdfsArrowADF.approxNRows)
    print(bigHdfsArrowADF)

    print(bigHdfsArrowADF.nRows)
    print(bigHdfsArrowADF)

    print(bigHdfsArrowADF.reprSample.shape)


    bigHdfsSparkADF = \
        SparkADF.load(
            path=BIG_DATA_HDFS_PATH,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(bigHdfsSparkADF)

    print(bigHdfsSparkADF.nRows)
    print(bigHdfsSparkADF)

    # print(bigHdfsADF.reprSample.shape)   # TOO SLOW


    bigHdfsArrowSparkADF = \
        ArrowSparkADF(
            path=BIG_DATA_HDFS_PATH,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(bigHdfsArrowSparkADF)

    print(bigHdfsArrowSparkADF.nRows)
    print(bigHdfsArrowSparkADF)

    print(bigHdfsArrowSparkADF.reprSample.shape)


    bigS3SparkADF = \
        SparkADF.load(
            path=BIG_DATA_S3_PATH,
            aws_access_key_id=_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(bigS3SparkADF)

    print(bigS3SparkADF.nRows)
    print(bigS3SparkADF)

    # print(bigS3ADF.reprSample.shape)   # TOO SLOW


    bigS3ArrowSparkADF = \
        ArrowSparkADF(
            path=BIG_DATA_S3_PATH,
            aws_access_key_id=_AWS_ACCESS_KEY_ID,
            aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
            iCol=EQUIPMENT_INSTANCE_ID_COL_NAME,
            tCol=DATE_TIME_COL_NAME)
    print(bigS3ArrowSparkADF)

    print(bigS3ArrowSparkADF.nRows)
    print(bigS3ArrowSparkADF)

    print(bigS3ArrowSparkADF.reprSample.shape)


import arimo.backend
print(arimo.backend.spark.conf.get('spark.files.maxPartitionBytes'),
      arimo.backend.spark.conf.get('spark.sql.files.maxPartitionBytes'),
      arimo.backend.spark.conf.get('spark.files.openCostInBytes'),
      arimo.backend.spark.conf.get('spark.sql.files.openCostInBytes'))
