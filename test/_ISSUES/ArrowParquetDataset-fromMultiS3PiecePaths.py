import os
import sys

from pyarrow.parquet import ParquetDataset
from s3fs import S3FileSystem

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data import _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY, SMALL_DATA_S3_PATH


fs = S3FileSystem(key=_AWS_ACCESS_KEY_ID, secret=_AWS_SECRET_ACCESS_KEY)


ds0 = ParquetDataset(
    path_or_paths=SMALL_DATA_S3_PATH,
    filesystem=fs,
    schema=None, validate_schema=False, metadata=None,
    split_row_groups=False)


ds1 = ParquetDataset(
    path_or_paths=list(piece.path for piece in ds0.pieces[:2]),
    filesystem=fs,
    schema=None, validate_schema=False, metadata=None,
    split_row_groups=False)
