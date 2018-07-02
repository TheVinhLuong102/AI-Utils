from __future__ import division, print_function

import datetime
import itertools
import pandas

from pyspark.sql.functions import to_date, spark_partition_id

import arimo.backend
from arimo.df.from_files import ArrowADF


ID_COL_NAME = 'id'
TIME_COL_NAME = 't'
CONTENT_COL_NAME = 'x'

DATE_COL_NAME = 'date'

_PARTITION_ID_COL = '__PARTITION_ID__'


YEAR = datetime.date.today().year


PARQUET_PATH_PREFIX = \
    '{}/tmp/TimeSerSparkADF'.format('hdfs:' if arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS else '')

PARQUET_PATH___PARTITITIONED_BY_DATE = \
    '{}-partitionBy-Date'.format(PARQUET_PATH_PREFIX)

PARQUET_PATH___REPARTITIONED___SORTED_WITHIN_PARTITIONS___PARTITIONED_BY_DATE = \
    '{}-repartitioned-sortedWithinPartitions-partitionBy-Date'.format(PARQUET_PATH_PREFIX)


arimo.backend.initSpark()


pandasDF = \
    pandas.DataFrame(
        data={
            ID_COL_NAME:
                9 * ['i2'] + 9 * ['i1'] + 9 * ['i0'],

            TIME_COL_NAME:
                3 * ['{} {}'.format(date_str, time_str)
                     for date_str, time_str in
                        itertools.product(
                            ['{}-01-03'.format(YEAR), '{}-01-02'.format(YEAR), '{}-01-01'.format(YEAR)],
                            ['18:18:18', '12:12:12', '06:06:06'])],

            CONTENT_COL_NAME:
                list(reversed(range(27)))}) \
    .sample(frac=1)   # randomize order

print(pandasDF)


sparkDF = \
    arimo.backend.spark.createDataFrame(data=pandasDF) \
    .withColumn(DATE_COL_NAME, to_date(TIME_COL_NAME)) \

sparkDF.cache()
n = sparkDF.count()

print('Orig Spark DF:')
sparkDF.withColumn(_PARTITION_ID_COL, spark_partition_id()).show(n)


sparkDF_sorted = sparkDF.sort(ID_COL_NAME, TIME_COL_NAME)

print('Spark DF Sorted:')
sparkDF_sorted.withColumn(_PARTITION_ID_COL, spark_partition_id()).show(n)


sparkDF_sorted_repartitioned = sparkDF_sorted.repartition(DATE_COL_NAME)

print('Spark DF Sorted then Repartitioned: ORDERING LOST:')
sparkDF_sorted_repartitioned.withColumn(_PARTITION_ID_COL, spark_partition_id()).show(n)

sparkDF_sorted_repartitioned.write.partitionBy(DATE_COL_NAME).parquet(PARQUET_PATH___PARTITITIONED_BY_DATE, mode='overwrite')
arrowADF_sorted_repartitioned = ArrowADF(path=PARQUET_PATH___PARTITITIONED_BY_DATE)
pandasDF_sorted_repartitioned = arrowADF_sorted_repartitioned.collect()
print(pandasDF_sorted_repartitioned)


sparkDF_repartitioned_sortedWithinPartitions = \
    sparkDF.repartition(DATE_COL_NAME).sortWithinPartitions(ID_COL_NAME, TIME_COL_NAME)

print('Spark DF Repartitioned then Sorted Within Partitions:')
sparkDF_repartitioned_sortedWithinPartitions.withColumn(_PARTITION_ID_COL, spark_partition_id()).show(n)

sparkDF_repartitioned_sortedWithinPartitions.write.partitionBy(DATE_COL_NAME).parquet(PARQUET_PATH___REPARTITIONED___SORTED_WITHIN_PARTITIONS___PARTITIONED_BY_DATE, mode='overwrite')
arrowADF_repartitioned_sortedWithinPartitions = ArrowADF(path=PARQUET_PATH___REPARTITIONED___SORTED_WITHIN_PARTITIONS___PARTITIONED_BY_DATE)
pandasDF_repartitioned_sortedWithinPartitions = arrowADF_repartitioned_sortedWithinPartitions.collect()
print(pandasDF_repartitioned_sortedWithinPartitions)
