import sys

from .distributed import DDF
from .distributed_parquet import S3ParquetDistributedDataFrame
from .parquet import S3ParquetDataFeeder

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'DDF',
    'S3ParquetDistributedDataFrame',
    'S3ParquetDataFeeder',
)
