from __future__ import division, print_function

import argparse
import os
import sys
import time

import arimo.backend
from arimo.util import fs, aws

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import \
    BIG_DATA_S3_PARQUET_PATH, BIG_DATA_S3A_AUTH_PARQUET_PATH, \
    BIG_DATA_S3_RAND_PARTITIONED_PARQUET_PATH, BIG_DATA_S3A_AUTH_RAND_PARTITIONED_PARQUET_PATH, \
    BIG_DATA_S3_ORC_PATH, BIG_DATA_S3A_AUTH_ORC_PATH, \
    BIG_DATA_S3_RAND_PARTITIONED_ORC_PATH, BIG_DATA_S3A_AUTH_RAND_PARTITIONED_ORC_PATH, \
    BIG_DATA_HDFS_PARQUET_PATH, BIG_DATA_RAND_PARTITIONED_HDFS_PARQUET_PATH, \
    BIG_DATA_HDFS_ORC_PATH, BIG_DATA_RAND_PARTITIONED_HDFS_ORC_PATH


key, secret = aws.key_pair('PanaAP-CC')


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--from-fs', default='s3a')
arg_parser.add_argument('--from-fmt', default='parquet')
arg_parser.add_argument('--from-rand', action='store_true')
arg_parser.add_argument('--to-fs', default='s3')
arg_parser.add_argument('--to-fmt', default='parquet')
arg_parser.add_argument('--to-rand', action='store_true')
arg_parser.add_argument('--spark', nargs='*', default=[])
args = arg_parser.parse_args()
print('*** CLI ARGS: ***\n{}\n'.format(args))


if args.from_fs == 's3a':
    if args.from_fmt == 'parquet':
        from_path = \
            BIG_DATA_S3A_AUTH_RAND_PARTITIONED_PARQUET_PATH \
            if args.from_rand \
            else BIG_DATA_S3A_AUTH_PARQUET_PATH

    else:
        assert args.from_fmt == 'orc'

        from_path = \
            BIG_DATA_S3A_AUTH_RAND_PARTITIONED_ORC_PATH \
            if args.from_rand \
            else BIG_DATA_S3A_AUTH_ORC_PATH

else:
    assert args.from_fs == 'hdfs'

    if args.from_fmt == 'parquet':
        from_path = \
            BIG_DATA_RAND_PARTITIONED_HDFS_PARQUET_PATH \
            if args.from_rand \
            else BIG_DATA_HDFS_PARQUET_PATH

    else:
        assert args.from_fmt == 'orc'

        from_path = \
            BIG_DATA_RAND_PARTITIONED_HDFS_ORC_PATH \
            if args.from_rand \
            else BIG_DATA_HDFS_ORC_PATH


if args.to_fs == 's3':
    args.to_fs = 'hdfs'

    if args.to_fmt == 'parquet':
        to_s3_path = \
            BIG_DATA_S3_RAND_PARTITIONED_PARQUET_PATH \
            if args.to_rand \
            else BIG_DATA_S3_PARQUET_PATH

    else:
        assert args.to_fmt == 'orc'

        to_s3_path = \
            BIG_DATA_S3_RAND_PARTITIONED_ORC_PATH \
            if args.to_rand \
            else BIG_DATA_S3_ORC_PATH

else:
    to_s3_path = None


if args.to_fs == 's3a':
    if args.to_fmt == 'parquet':
        to_path = \
            BIG_DATA_S3A_AUTH_RAND_PARTITIONED_PARQUET_PATH \
            if args.to_rand \
            else BIG_DATA_S3A_AUTH_PARQUET_PATH

    else:
        assert args.to_fmt == 'orc'

        to_path = \
            BIG_DATA_S3A_AUTH_RAND_PARTITIONED_ORC_PATH \
            if args.to_rand \
            else BIG_DATA_S3A_AUTH_ORC_PATH

else:
    assert args.to_fs == 'hdfs'

    if args.to_fmt == 'parquet':
        to_path = \
            BIG_DATA_RAND_PARTITIONED_HDFS_PARQUET_PATH \
            if args.to_rand \
            else BIG_DATA_HDFS_PARQUET_PATH

    else:
        assert args.to_fmt == 'orc'

        to_path = \
            BIG_DATA_RAND_PARTITIONED_HDFS_ORC_PATH \
            if args.to_rand \
            else BIG_DATA_HDFS_ORC_PATH


spark_conf = {}

for i in range(0, len(args.spark), 2):
    spark_conf[args.spark[i]] = args.spark[i + 1]

arimo.backend.init(
    sparkApp='PanaCC test Spark read/write big data: from {} to {}'
        .format(
            from_path,
            to_s3_path
                if to_s3_path
                else to_path),
    sparkConf=spark_conf)


print('LOADING FROM {}... '.format(from_path), end=' ')
sdf = arimo.backend.spark.read.load(
    path=from_path,
    format=args.from_fmt,
    schema=None)
print('done!')


print('SAVING TO {}... '.format(to_path), end=' ')
tic = time.time()
sdf.write.save(
    path=to_path,
    format=args.to_fmt,
    mode='overwrite',
    partitionBy=None
        if args.to_rand
        else 'date')
toc = time.time()
print('done! <{:,.0f} mins>'.format((toc - tic) / 60))


if to_s3_path:
    fs.get(
        from_hdfs=to_path,
        to_local=to_path,
        is_dir=True,
        overwrite=True,
        _mv=False,
        must_succeed=True)

    aws.s3_sync(
        from_dir_path=to_path,
        to_dir_path=to_s3_path,
        delete=True, quiet=True,
        access_key_id=key, secret_access_key=secret,
        verbose=True)
