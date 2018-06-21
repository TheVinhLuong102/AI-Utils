import time

import arimo.backend
from arimo.util.aws import key_pair


key, secret = key_pair('PanaAP-CC')


arimo.backend.initSpark()


sdf0 = arimo.backend.spark.read.parquet('s3a://{}:{}@arimo-panasonic-ap/data/CombinedConfigMeasure/1003_a01.parquet'.format(key, secret))
sdf0.cache(); sdf0.count()

tic = time.time()
sdf0.write.parquet('/tmp/test0.pqt', mode='overwrite')
toc = time.time()
print('HDFS w/o partitions: {:.0f}s'.format(toc - tic))

tic = time.time()
sdf0.write.parquet('s3a://{}:{}@arimo-panasonic-ap/tmp/test0.pqt'.format(key, secret), mode='overwrite')
toc = time.time()
print('S3 w/o partitions: {:.0f}s'.format(toc - tic))


sdf1 = sdf0

tic = time.time()
sdf1.write.parquet('/tmp/test1.pqt', partitionBy='date', mode='overwrite')
toc = time.time()
print('HDFS w partitionBy: {:.0f}s'.format(toc - tic))

tic = time.time()
sdf1.write.parquet('s3a://{}:{}@arimo-panasonic-ap/tmp/test1.pqt'.format(key, secret), partitionBy='date', mode='overwrite')
toc = time.time()
print('S3 w partitionBy: {:.0f}s'.format(toc - tic))


sdf2 = sdf1.repartition('date')
sdf2.cache(); sdf2.count()

tic = time.time()
sdf2.write.parquet('/tmp/test2.pqt', partitionBy='date', mode='overwrite')
toc = time.time()
print('HDFS w repartition & partitionBy: {:.0f}s'.format(toc - tic))

tic = time.time()
sdf2.write.parquet('s3a://{}:{}@arimo-panasonic-ap/tmp/test2.pqt'.format(key, secret), partitionBy='date', mode='overwrite')
toc = time.time()
print('S3 w repartition & partitionBy: {:.0f}s'.format(toc - tic))
