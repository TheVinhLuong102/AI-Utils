import sys

from .parquet import S3ParquetDataFeeder

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence


__all__: Sequence[str] = (
    'S3ParquetDataFeeder',
)
