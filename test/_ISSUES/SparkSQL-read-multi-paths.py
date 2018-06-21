import os
import sys

import arimo.backend

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import BIG_DATA_S3A_AUTH_PARQUET_PATH, BIG_DATA_S3A_AUTH_RAND_PARTITIONED_PARQUET_PATH


arimo.backend.initSpark()


# assertion failed: Conflicting directory structures detected. Suspicious paths: ...
# If provided paths are partition directories,
# please set "basePath" in the options of the data source to specify the root directory of the table.
# If there are multiple root directories, please load them separately and then union them.
sdf = arimo.backend.spark.read.parquet(BIG_DATA_S3A_AUTH_PARQUET_PATH, BIG_DATA_S3A_AUTH_RAND_PARTITIONED_PARQUET_PATH)
