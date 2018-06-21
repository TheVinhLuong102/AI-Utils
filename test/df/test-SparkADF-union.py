import os

import arimo.df
from arimo.df.spark import SparkADF
from arimo.util import fs


RESOURCES_DIR_PATH = \
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                arimo.backend.__file__)),
        'resources')


TINY_PARQUET_PATH = \
    os.path.join(
        RESOURCES_DIR_PATH,
        'tiny.parquet')

if fs._ON_LINUX_CLUSTER_WITH_HDFS:
    fs.put(
        from_local=TINY_PARQUET_PATH,
        to_hdfs=TINY_PARQUET_PATH,
        is_dir=False,
        _mv=False)


SKEWED_DATA_PARQUET_PATH = \
    os.path.join(
        RESOURCES_DIR_PATH,
        'skewed-data.parquet')

fs.put(
    from_local=SKEWED_DATA_PARQUET_PATH,
    to_hdfs=SKEWED_DATA_PARQUET_PATH,
    is_dir=True,
    _mv=False)


adf_0 = SparkADF.load(
    path=TINY_PARQUET_PATH,
    format='parquet',
    verbose=True)


adf_1 = SparkADF.load(
    path=SKEWED_DATA_PARQUET_PATH,
    format='parquet',
    verbose=True)


adf_0_0_a = SparkADF.unionAllCols(adf_0, adf_0)
assert adf_0_0_a.nPartitions == 2 * adf_0.nPartitions

adf_0_0_b = SparkADF.unionAllCols(adf_0, adf_0, _unionRDDs=True)
assert adf_0_0_b.nPartitions == 2 * adf_0.nPartitions


adf_0_1_a = SparkADF.unionAllCols(adf_0, adf_1)
assert adf_0_1_a.nPartitions == adf_0.nPartitions + adf_1.nPartitions

adf_0_1_b = SparkADF.unionAllCols(adf_0, adf_1, _unionRDDs=True)
assert adf_0_1_b.nPartitions == adf_0.nPartitions + adf_1.nPartitions


adf_1_1_a = SparkADF.unionAllCols(adf_1, adf_1)
assert adf_1_1_a.nPartitions == 2 * adf_1.nPartitions

adf_1_1_b = SparkADF.unionAllCols(adf_1, adf_1, _unionRDDs=True)
assert adf_1_1_b.nPartitions == 2 * adf_1.nPartitions
