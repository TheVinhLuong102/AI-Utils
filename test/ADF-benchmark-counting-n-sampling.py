from __future__ import division, print_function

import os
import pandas
import time

import arimo.backend
from arimo.df import ADF
from arimo.dl.reader import S3ParquetDatasetReader

from PanasonicColdChain import _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY


os.environ['AWS_ACCESS_KEY_ID'] = _AWS_ACCESS_KEY_ID
os.environ['AWS_SECRET_ACCESS_KEY'] = _AWS_SECRET_ACCESS_KEY


PARQUET_PATH = 's3://arimo-panasonic-ap/data/CombinedConfigMeasure/DISP_CASE---ex_display_case.parquet'
_PARQUET_PATH = 's3a://{}:{}@arimo-panasonic-ap/tmp/prep.pqt'.format(_AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY)

ID_COL = 'equipment_instance_id'
DATE_TIME_COL = 'date_time'


DATE_FROM = '2015-01-01'
DATE_TO = '2015-12-31'


FILTER_CONDITION = '{} IS NOT NULL'.format(DATE_TIME_COL)


SORT_SQL_STATEMENT = \
    'SELECT \
        *, \
        ROW_NUMBER() OVER (PARTITION BY {} ORDER BY {}) AS __tOrd__ \
    FROM this'.format(
        ID_COL, DATE_TIME_COL)


arimo.backend.init(sparkConf={'spark.executor.memory': '9g'})


_adf = ADF.load(
        path=PARQUET_PATH,
        aws_access_key_id=_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=_AWS_SECRET_ACCESS_KEY) \
    .filter(
        "({0} >= '{1}') AND ({0} <= '{2}')".format(
            ADF._DEFAULT_D_COL,
            DATE_FROM, DATE_TO))



adf = _adf.rm(ADF._DEFAULT_D_COL)


filtered_adf = adf.filter(FILTER_CONDITION)


sorted_adf = adf(SORT_SQL_STATEMENT)


sorted_filtered_adf = filtered_adf(SORT_SQL_STATEMENT)


id_partitioned_adf = adf.repartition(ID_COL)
id_n_date_partitioned_adf = _adf.repartition(ID_COL, ADF._DEFAULT_D_COL)


t_adf = adf('*', tCol=DATE_TIME_COL)
ts_adf = adf('*', iCol=ID_COL, tCol=DATE_TIME_COL)


tic = time.time()
count = adf._sparkDF.count()
toc = time.time()
count_time = toc - tic
print('COUNT: {:,} <{:.1f}s>'.format(count, count_time))


tic = time.time()
filtered_count = filtered_adf._sparkDF.count()
toc = time.time()
filtered_count_time = toc - tic


tic = time.time()
sorted_count = sorted_adf._sparkDF.count()
toc = time.time()
sorted_count_time = toc - tic


tic = time.time()
sorted_filtered_count = sorted_filtered_adf._sparkDF.count()
toc = time.time()
sorted_filtered_count_time = toc - tic


tic = time.time()
id_partitioned_count = id_partitioned_adf._sparkDF.count()
toc = time.time()
id_partitioned_count_time = toc - tic


tic = time.time()
id_n_date_partitioned_count = id_n_date_partitioned_adf._sparkDF.count()
toc = time.time()
id_n_date_partitioned_count_time = toc - tic


tic = time.time()
t_adf_count = t_adf._sparkDF.count()
toc = time.time()
t_adf_count_time = toc - tic


tic = time.time()
ts_adf_count = ts_adf._sparkDF.count()
toc = time.time()
ts_adf_count_time = toc - tic
print('TS ADF COUNT: {:,} <{:.1f}s>'.format(ts_adf_count, ts_adf_count_time))


ts_sdf = ts_adf._sparkDF
ts_sdf.write.parquet('/tmp/testprep.pqt', mode='overwrite')


metrics_df = \
    pandas.DataFrame(
        index=['count'],
        columns=['_',
                 'filtered',
                 'sorted',
                 'sorted_filtered',
                 'id_partitioned',
                 'id_n_date_partitioned'])

metrics_df.at['count', '_'] = count_time
metrics_df.at['count', 'filtered'] = filtered_count_time
metrics_df.at['count', 'sorted'] = sorted_count_time
metrics_df.at['count', 'sorted_filtered'] = sorted_filtered_count_time
metrics_df.at['count', 'id_partitioned'] = id_partitioned_count_time
metrics_df.at['count', 'id_n_date_partitioned'] = id_n_date_partitioned_count_time

metrics_df


import pyarrow.parquet as pq
import s3fs

dataset = pq.ParquetDataset('s3://arimo-panasonic-ap/data/CombinedConfigMeasure/DISP_CASE---ex_display_case.parquet',
                            filesystem=s3fs.S3FileSystem(key=_AWS_ACCESS_KEY_ID, secret=_AWS_SECRET_ACCESS_KEY))


_adf.save(
    path='s3://arimo-panasonic-ap/data/CombinedConfigMeasure/DISP_CASE---ex_display_case.orc',
    format='orc',
    partitionBy='date',
    aws_access_key_id=_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=_AWS_SECRET_ACCESS_KEY)
