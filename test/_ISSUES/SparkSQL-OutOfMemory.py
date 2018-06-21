from __future__ import print_function

import argparse
import gc
import pandas

import pyspark


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--unpersist', action='store_true')
arg_parser.add_argument('--py-gc', action='store_true')
arg_parser.add_argument('--n-partitions', type=int, default=1000)
args = arg_parser.parse_args()


# create SparkSession (*** set spark.driver.memory to 512m in spark-defaults.conf ***)
spark = pyspark.sql.SparkSession.builder \
    .config('spark.executor.instances', 2) \
    .config('spark.executor.cores', 2) \
    .config('spark.executor.memory', '512m') \
    .config('spark.ui.enabled', False) \
    .config('spark.ui.retainedJobs', 10) \
    .config('spark.ui.retainedStages', 10) \
    .config('spark.ui.retainedTasks', 10) \
    .enableHiveSupport() \
    .getOrCreate()


# create Parquet file to subsequent repeated loading
df = spark.createDataFrame(
    pandas.DataFrame(
        dict(
            row=range(args.n_partitions),
            x=args.n_partitions * [0]
        )
    )
)

parquet_path = '/tmp/TestOOM-{}Partitions.parquet'.format(args.n_partitions)

df.write.parquet(
    path=parquet_path,
    partitionBy='row',
    mode='overwrite'
)


i = 0

while True:
    _df = spark.read.parquet(parquet_path)

    if args.unpersist:
        _df.unpersist()

    if args.py_gc:
        del _df
        gc.collect()

    i += 1; print('COMPLETED READ ITERATION #{}\n'.format(i))
