"""S3 Parquet Data Feeder."""


from __future__ import annotations

import datetime
from functools import lru_cache, partial
from itertools import chain
from logging import Logger
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Optional, Union
from typing import Collection, Dict, List, Set, Sequence, Tuple   # Py3.9+: use built-ins
from urllib.parse import ParseResult, urlparse
from uuid import uuid4

from numpy import isfinite, ndarray, vstack
from pandas import DataFrame, Series, concat, isnull, notnull, read_parquet
from pandas._libs.missing import NA, NAType   # pylint: disable=no-name-in-module
from tqdm import tqdm

from pyarrow.dataset import dataset
from pyarrow.fs import S3FileSystem
from pyarrow.lib import RecordBatch, Schema, Table   # pylint: disable=no-name-in-module
from pyarrow.parquet import FileMetaData, read_metadata, read_schema, read_table

from .. import debug, s3
from ..data_types.arrow import (
    DataType, _ARROW_STR_TYPE, _ARROW_DATE_TYPE,
    is_binary, is_boolean, is_num, is_possible_cat, is_possible_feature, is_string)
from ..data_types.numpy_pandas import NUMPY_FLOAT_TYPES, NUMPY_INT_TYPES
from ..data_types.python import PY_NUM_TYPES, PyNumType, PyPossibleFeatureType, PY_LIST_OR_TUPLE
from ..default_dict import DefaultDict
from ..fs import PathType, mkdir
from ..iter import to_iterable
from ..namespace import Namespace, DICT_OR_NAMESPACE_TYPES

from ._abstract import (AbstractDataHandler,
                        AbstractFileDataHandler,
                        AbstractS3FileDataHandler,
                        ReducedDataSetType)
from .pandas import PandasMLPreprocessor


__all__ = ('S3ParquetDataFeeder',)


# flake8: noqa
# (too many camelCase names)

# pylint: disable=c-extension-no-member
# e.g., `math.`

# pylint: disable=invalid-name
# e.g., camelCase names

# pylint: disable=no-member
# e.g., `._cache`

# pylint: disable=protected-access
# e.g., `._cache`

# pylint: disable=too-many-lines
# (this whole module)


CallablesType = Union[callable, Sequence[callable]]
ColsType = Union[str, Collection[str]]


def randomSample(population: Collection[Any], sampleSize: int,
                 returnCollectionType=set) -> Collection[Any]:
    """Draw random sample from population."""
    return returnCollectionType(random.sample(population=population, k=sampleSize)
                                if len(population) > sampleSize
                                else population)


class S3ParquetDataFeeder(AbstractS3FileDataHandler):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """S3 Parquet Data Feeder."""

    # caches
    _CACHE: Dict[str, Namespace] = {}
    _FILE_CACHES: Dict[str, Namespace] = {}

    # default arguments dict
    # (cannot be h1st_util.namespace.Namespace
    # because that makes nested dicts into normal dicts)
    _DEFAULT_KWARGS: Dict[str, Optional[Union[str, DefaultDict]]] = dict(
        iCol=None, tCol=None,

        reprSampleMinNFiles=AbstractFileDataHandler._REPR_SAMPLE_MIN_N_FILES,
        reprSampleSize=AbstractDataHandler._DEFAULT_REPR_SAMPLE_SIZE,

        nulls=DefaultDict((None, None)),
        minNonNullProportion=DefaultDict(AbstractDataHandler._DEFAULT_MIN_NON_NULL_PROPORTION),
        outlierTailProportion=DefaultDict(AbstractDataHandler._DEFAULT_OUTLIER_TAIL_PROPORTION),
        maxNCats=DefaultDict(AbstractDataHandler._DEFAULT_MAX_N_CATS),
        minProportionByMaxNCats=DefaultDict(
            AbstractDataHandler._DEFAULT_MIN_PROPORTION_BY_MAX_N_CATS))

    def __init__(self, path: str, *, awsRegion: Optional[str] = None,
                 _mappers: Optional[CallablesType] = None,
                 _reduceMustInclCols: Optional[ColsType] = None,
                 verbose: bool = True, **kwargs: Any):
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Init S3 Parquet Data Feeder."""
        if verbose or debug.ON:
            logger: Logger = self.classStdOutLogger()

        assert isinstance(path, str) and path.startswith('s3://'), \
            ValueError(f'*** {path} NOT AN S3 PATH ***')
        self.path: str = path

        self.awsRegion: Optional[str] = awsRegion

        if path in self._CACHE:
            _cache: Namespace = self._CACHE[path]
        else:
            self._CACHE[path] = _cache = Namespace()

        if _cache:
            if debug.ON:
                logger.debug(msg=f'*** RETRIEVING CACHE FOR "{path}" ***')

        else:
            _parsedURL: ParseResult = urlparse(url=path, scheme='', allow_fragments=True)
            _cache.s3Bucket = _parsedURL.netloc
            _cache.pathS3Key = _parsedURL.path[1:]

            _cache.tmpDirPath = f's3://{_cache.s3Bucket}/{self._TMP_DIR_S3_KEY}'

            if path in self._FILE_CACHES:
                _cache.nFiles = 1
                _cache.filePaths = {path}

            else:
                if verbose:
                    logger.info(msg=(msg := f'Loading "{path}" by Arrow...'))
                    tic: float = time.time()

                s3.rm(path=path,
                      is_dir=True,
                      globs='*_$folder$',   # redundant AWS EMR-generated files
                      quiet=True,
                      verbose=False)

                _cache._srcArrowDS = \
                    dataset(source=path.replace('s3://', ''),
                            schema=None,
                            format='parquet',
                            filesystem=S3FileSystem(region=awsRegion),
                            partitioning=None,
                            partition_base_dir=None,
                            exclude_invalid_files=None,
                            ignore_prefixes=None)

                if verbose:
                    toc: float = time.time()
                    logger.info(msg=f'{msg} done!   <{toc - tic:,.1f} s>')

                if _filePaths := _cache._srcArrowDS.files:
                    _cache.filePaths = {f's3://{filePath}'
                                        for filePath in _filePaths
                                        if not filePath.endswith('_$folder$')}
                    _cache.nFiles = len(_cache.filePaths)

                else:
                    _cache.nFiles = 1
                    _cache.filePaths = {path}

            _cache.srcColsInclPartitionKVs = set()
            _cache.srcTypesInclPartitionKVs = Namespace()

            for i, filePath in enumerate(_cache.filePaths):
                if filePath in self._FILE_CACHES:
                    fileCache: Namespace = self._FILE_CACHES[filePath]

                    if (fileCache.nRows is None) and (i < self._SCHEMA_MIN_N_FILES):
                        fileCache.localPath = self.fileLocalPath(filePath=filePath)

                        schema: Schema = read_schema(where=fileCache.localPath)

                        fileCache.srcColsExclPartitionKVs = set(schema.names)

                        fileCache.srcColsInclPartitionKVs.update(schema.names)

                        for col in (fileCache.srcColsExclPartitionKVs
                                    .difference(fileCache.partitionKVs)):
                            fileCache.srcTypesExclPartitionKVs[col] = \
                                fileCache.srcTypesInclPartitionKVs[col] = \
                                schema.field(col).type

                        metadata: FileMetaData = read_metadata(where=fileCache.localPath)
                        fileCache.nCols = metadata.num_columns
                        fileCache.nRows = metadata.num_rows

                else:
                    srcColsInclPartitionKVs: Set[str] = set()

                    srcTypesExclPartitionKVs: Namespace = Namespace()
                    srcTypesInclPartitionKVs: Namespace = Namespace()

                    partitionKVs: Dict[str, Union[datetime.date, str]] = {}

                    for partitionKV in re.findall(pattern='[^/]+=[^/]+/', string=filePath):
                        k, v = partitionKV.split(sep='=', maxsplit=1)

                        srcColsInclPartitionKVs.add(k)

                        if k == self._DATE_COL:
                            srcTypesInclPartitionKVs[k] = _ARROW_DATE_TYPE
                            partitionKVs[k] = datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date()

                        else:
                            srcTypesInclPartitionKVs[k] = _ARROW_STR_TYPE
                            partitionKVs[k] = v[:-1]

                    if i < self._SCHEMA_MIN_N_FILES:
                        localPath: Path = self.fileLocalPath(filePath=filePath)

                        schema: Schema = read_schema(where=localPath)

                        srcColsExclPartitionKVs: Set[str] = set(schema.names)

                        srcColsInclPartitionKVs.update(schema.names)

                        for col in srcColsExclPartitionKVs.difference(partitionKVs):
                            srcTypesExclPartitionKVs[col] = \
                                srcTypesInclPartitionKVs[col] = \
                                schema.field(col).type

                        metadata: FileMetaData = read_metadata(where=localPath)
                        nCols: int = metadata.num_columns
                        nRows: int = metadata.num_rows

                    else:
                        localPath: Optional[Path] = None

                        srcColsExclPartitionKVs: Optional[Set[str]] = None

                        nCols: Optional[int] = None
                        nRows: Optional[int] = None

                    self._FILE_CACHES[filePath] = fileCache = \
                        Namespace(localPath=localPath,

                                  partitionKVs=partitionKVs,

                                  srcColsExclPartitionKVs=srcColsExclPartitionKVs,
                                  srcColsInclPartitionKVs=srcColsInclPartitionKVs,

                                  srcTypesExclPartitionKVs=srcTypesExclPartitionKVs,
                                  srcTypesInclPartitionKVs=srcTypesInclPartitionKVs,

                                  nCols=nCols, nRows=nRows)

                _cache.srcColsInclPartitionKVs |= fileCache.srcColsInclPartitionKVs

                for col, arrowType in fileCache.srcTypesInclPartitionKVs.items():
                    if col in _cache.srcTypesInclPartitionKVs:
                        assert arrowType == _cache.srcTypesInclPartitionKVs[col], \
                            TypeError(f'*** {filePath} COLUMN {col}: '
                                      f'DETECTED TYPE {arrowType} != '
                                      f'{_cache.srcTypesInclPartitionKVs[col]} ***')
                    else:
                        _cache.srcTypesInclPartitionKVs[col] = arrowType

            _cache.cachedLocally = False

        self.__dict__.update(_cache)

        self._mappers: Tuple[callable] = (()
                                          if _mappers is None
                                          else to_iterable(_mappers, iterable_type=tuple))

        self._reduceMustInclCols: Set[str] = (set()
                                              if _reduceMustInclCols is None
                                              else to_iterable(_reduceMustInclCols,
                                                               iterable_type=set))

        # extract standard keyword arguments
        self._extractStdKwArgs(kwargs, resetToClassDefaults=True, inplace=True)

        # organize time series if applicable
        self._organizeIndexCols()

        # set profiling settings and create empty profiling cache
        self._emptyCache()

    # ===========
    # STRING REPR
    # -----------
    # __repr__
    # __shortRepr__

    def __repr__(self) -> str:
        """Return string repr."""
        colAndTypeStrs: List[str] = []

        if self._iCol:
            colAndTypeStrs.append(f'(iCol) {self._iCol}: {self.type(self._iCol)}')

        if self._dCol:
            colAndTypeStrs.append(f'(dCol) {self._dCol}: {self.type(self._dCol)}')

        if self._tCol:
            colAndTypeStrs.append(f'(tCol) {self._tCol}: {self.type(self._tCol)}')

        colAndTypeStrs.extend(f'{col}: {self.type(col)}' for col in self.contentCols)

        return (f'{self.nFiles:,}-file ' +
                (f'{self._cache.nRows:,}-row '
                 if self._cache.nRows
                 else (f'approx-{self._cache.approxNRows:,.0f}-row '
                       if self._cache.approxNRows
                       else '')) +
                type(self).__name__ +
                (f'[{self.path} + {len(self._mappers):,} transform(s)]'
                 f"[{', '.join(colAndTypeStrs)}]"))

    @property
    def __shortRepr__(self) -> str:
        """Short string repr."""
        colsDescStr: List[str] = []

        if self._iCol:
            colsDescStr.append(f'iCol: {self._iCol}')

        if self._dCol:
            colsDescStr.append(f'dCol: {self._dCol}')

        if self._tCol:
            colsDescStr.append(f'tCol: {self._tCol}')

        colsDescStr.append(f'{len(self.contentCols)} content col(s)')

        return (f'{self.nFiles:,}-file ' +
                (f'{self._cache.nRows:,}-row '
                 if self._cache.nRows
                 else (f'approx-{self._cache.approxNRows:,.0f}-row '
                       if self._cache.approxNRows
                       else '')) +
                type(self).__name__ +
                (f'[{self.path} + {len(self._mappers):,} transform(s)]'
                 f"[{', '.join(colsDescStr)}]"))

    # ================================
    # "INTERNAL / DON'T TOUCH" METHODS
    # --------------------------------
    # _extractStdKwArgs
    # _organizeIndexCols

    # pylint: disable=inconsistent-return-statements
    def _extractStdKwArgs(self, kwargs: Dict[str, Any], /, *,
                          resetToClassDefaults: bool = False,
                          inplace: bool = False) -> Optional[Namespace]:
        namespace: Union[S3ParquetDataFeeder, Namespace] = self if inplace else Namespace()

        for k, classDefaultV in self._DEFAULT_KWARGS.items():
            _privateK: str = f'_{k}'

            if not resetToClassDefaults:
                existingInstanceV: Any = getattr(self, _privateK, None)

            v: Any = kwargs.pop(k,
                                existingInstanceV
                                if (not resetToClassDefaults) and existingInstanceV
                                else classDefaultV)

            if (k == 'reprSampleMinNFiles') and (v > self.nFiles):
                v: int = self.nFiles

            setattr(namespace,
                    _privateK   # use _k to not invoke @k.setter right away
                    if inplace
                    else k,
                    v)

        if inplace:
            if self._iCol not in self.columns:
                self._iCol: Optional[str] = None

            if self._tCol not in self.columns:
                self._tCol: Optional[str] = None

        else:
            return namespace

    def _organizeIndexCols(self):
        self._dCol: Optional[str] = self._DATE_COL if self._DATE_COL in self.columns else None

    # =======
    # CACHING
    # -------
    # _emptyCache
    # _inheritCache
    # cacheLocally
    # fileLocalPath
    # _readMetadataAndSchema

    def _emptyCache(self):
        self._cache: Namespace = \
            Namespace(prelimReprSampleFilePaths=None,
                      reprSampleFilePaths=None,
                      reprSample=None,

                      approxNRows=None, nRows=None,

                      count={}, distinct={},

                      nonNullProportion={},
                      suffNonNullProportionThreshold={}, suffNonNull={},

                      sampleMin={}, sampleMax={},
                      sampleMean={}, sampleMedian={},

                      outlierRstMin={}, outlierRstMax={},
                      outlierRstMean={}, outlierRstMedian={})

    def _inheritCache(self, oldS3ParquetDF: S3ParquetDataFeeder, /,
                      *sameCols: str, **newColToOldColMap: str):
        # pylint: disable=arguments-differ
        if oldS3ParquetDF._cache.nRows:
            if self._cache.nRows is None:
                self._cache.nRows = oldS3ParquetDF._cache.nRows
            else:
                assert self._cache.nRows == oldS3ParquetDF._cache.nRows

        if oldS3ParquetDF._cache.approxNRows and (self._cache.approxNRows is None):
            self._cache.approxNRows = oldS3ParquetDF._cache.approxNRows

        commonCols: Set[str] = self.columns.intersection(oldS3ParquetDF.columns)

        if sameCols or newColToOldColMap:
            for newCol, oldCol in newColToOldColMap.items():
                assert newCol in self.columns
                assert oldCol in oldS3ParquetDF.columns

            for sameCol in commonCols.difference(newColToOldColMap).intersection(sameCols):
                newColToOldColMap[sameCol] = sameCol

        else:
            newColToOldColMap: Dict[str, str] = {col: col for col in commonCols}

        for cacheCategory in (
                'count', 'distinct',
                'nonNullProportion', 'suffNonNullProportionThreshold', 'suffNonNull',
                'sampleMin', 'sampleMax', 'sampleMean', 'sampleMedian',
                'outlierRstMin', 'outlierRstMax', 'outlierRstMean', 'outlierRstMedian'):
            for newCol, oldCol in newColToOldColMap.items():
                if oldCol in oldS3ParquetDF._cache.__dict__[cacheCategory]:
                    self._cache.__dict__[cacheCategory][newCol] = \
                        oldS3ParquetDF._cache.__dict__[cacheCategory][oldCol]

    def cacheLocally(self, verbose: bool = True):
        """Cache files to local disk."""
        if not (_cache := self._CACHE[self.path]).cachedLocally:
            if verbose:
                self.stdOutLogger.info(msg=(msg := 'Caching Files to Local Disk...'))
                tic: float = time.time()

            parsedURL: ParseResult = urlparse(url=self.path, scheme='', allow_fragments=True)

            localPath: str = str(self._LOCAL_CACHE_DIR_PATH / parsedURL.netloc / parsedURL.path[1:])

            s3.sync(from_dir_path=self.path, to_dir_path=localPath,
                    delete=True, quiet=True, verbose=True)

            for filePath in self.filePaths:
                self._FILE_CACHES[filePath].localPath = filePath.replace(self.path, localPath)

            _cache.cachedLocally = True

            if verbose:
                toc: float = time.time()
                self.stdOutLogger.info(msg=f'{msg} done!   <{toc - tic:,.1f} s>')

    def fileLocalPath(self, filePath: str) -> Path:
        """Get local cache file path."""
        if (filePath in self._FILE_CACHES) and self._FILE_CACHES[filePath].localPath:
            return self._FILE_CACHES[filePath].localPath

        parsedURL: ParseResult = urlparse(url=filePath, scheme='', allow_fragments=True)

        localPath: Path = self._LOCAL_CACHE_DIR_PATH / parsedURL.netloc / parsedURL.path[1:]

        localDirPath: Path = localPath.parent
        mkdir(dir_path=localDirPath, hdfs=False)
        # make sure the dir has been created
        while not localDirPath.is_dir():
            time.sleep(1)

        self.S3_CLIENT.download_file(Bucket=parsedURL.netloc,
                                     Key=parsedURL.path[1:],
                                     Filename=str(localPath))
        # make sure AWS S3's asynchronous process has finished
        # downloading a potentially large file
        while not localPath.is_file():
            time.sleep(1)

        if filePath in self._FILE_CACHES:
            self._FILE_CACHES[filePath].localPath = localPath

        return localPath

    def _readMetadataAndSchema(self, filePath: str) -> Namespace:
        fileLocalPath: Path = self.fileLocalPath(filePath=filePath)

        fileCache: Namespace = self._FILE_CACHES[filePath]

        if fileCache.nRows is None:
            schema: Schema = read_schema(where=fileLocalPath)

            fileCache.srcColsExclPartitionKVs = set(schema.names)

            fileCache.srcColsInclPartitionKVs.update(schema.names)

            self.srcColsInclPartitionKVs.update(schema.names)

            for col in fileCache.srcColsExclPartitionKVs.difference(fileCache.partitionKVs):
                fileCache.srcTypesExclPartitionKVs[col] = \
                    fileCache.srcTypesInclPartitionKVs[col] = \
                    _arrowType = schema.field(col).type

                assert not is_binary(_arrowType), \
                    TypeError(f'*** {filePath}: {col} IS OF BINARY TYPE ***')

                if col in self.srcTypesInclPartitionKVs:
                    assert _arrowType == self.srcTypesInclPartitionKVs[col], \
                        TypeError(f'*** {filePath} COLUMN {col}: '
                                  f'DETECTED TYPE {_arrowType} != '
                                  f'{self.srcTypesInclPartitionKVs[col]} ***')
                else:
                    self.srcTypesInclPartitionKVs[col] = _arrowType

            metadata: FileMetaData = read_metadata(where=fileCache.localPath)
            fileCache.nCols = metadata.num_columns
            fileCache.nRows = metadata.num_rows

        return fileCache

    # ====
    # COPY
    # ----
    # copy

    def copy(self, **kwargs: Any) -> S3ParquetDataFeeder:
        """Make a copy."""
        resetMappers: bool = kwargs.pop('resetMappers', False)
        resetReduceMustInclCols: bool = kwargs.pop('resetReduceMustInclCols', False)
        inheritCache: bool = kwargs.pop('inheritCache', not resetMappers)
        inheritNRows: bool = kwargs.pop('inheritNRows', inheritCache)

        s3ParquetDF: S3ParquetDataFeeder = \
            S3ParquetDataFeeder(
                path=self.path, awsRegion=self.awsRegion,

                iCol=self._iCol, tCol=self._tCol,

                _mappers=() if resetMappers else self._mappers,
                _reduceMustInclCols=set() if resetReduceMustInclCols else self._reduceMustInclCols,

                reprSampleMinNFiles=self._reprSampleMinNFiles,
                reprSampleSize=self._reprSampleSize,

                minNonNullProportion=self._minNonNullProportion,
                outlierTailProportion=self._outlierTailProportion,
                maxNCats=self._maxNCats,
                minProportionByMaxNCats=self._minProportionByMaxNCats,

                **kwargs)

        if inheritCache:
            s3ParquetDF._inheritCache(self)

        if inheritNRows:
            s3ParquetDF._cache.approxNRows = self._cache.approxNRows
            s3ParquetDF._cache.nRows = self._cache.nRows

        return s3ParquetDF

    # =====================
    # ROWS, COLUMNS & TYPES
    # ---------------------
    # approxNRows / nRows / __len__
    # columns
    # indexCols
    # types
    # type / typeIsNum
    # possibleFeatureCols
    # possibleCatCols

    @property
    def approxNRows(self) -> int:
        """Approximate number of rows."""
        if self._cache.approxNRows is None:
            self.stdOutLogger.info(msg='Counting Approx. No. of Rows...')

            self._cache.approxNRows = (
                self.nFiles
                * sum(self._readMetadataAndSchema(filePath=filePath).nRows
                      for filePath in (tqdm(self.prelimReprSampleFilePaths)
                                       if len(self.prelimReprSampleFilePaths) > 1
                                       else self.prelimReprSampleFilePaths))
                / self._reprSampleMinNFiles)

        return self._cache.approxNRows

    @property
    def nRows(self) -> int:
        """Return number of rows."""
        if self._cache.nRows is None:
            self.stdOutLogger.info(msg='Counting No. of Rows...')

            self._cache.nRows = \
                sum(self._readMetadataAndSchema(filePath=filePath).nRows
                    for filePath in (tqdm(self.filePaths) if self.nFiles > 1 else self.filePaths))

        return self._cache.nRows

    def __len__(self) -> int:
        """Return (approximate) number of rows."""
        return self._cache.nRows if self._cache.nRows else self.approxNRows

    @property
    def columns(self) -> Set[str]:
        """Column names."""
        return self.srcColsInclPartitionKVs

    @property
    def indexCols(self) -> Set[str]:
        """Return index columns."""
        return (({self._iCol} if self._iCol else set()) |
                ({self._dCol} if self._dCol else set()) |
                ({self._tCol} if self._tCol else set()))

    @property
    def types(self) -> Namespace:
        """Return column data types."""
        return self.srcTypesInclPartitionKVs

    @lru_cache(maxsize=None, typed=False)
    def type(self, col: str) -> DataType:
        """Return data type of specified column."""
        return self.types[col]

    @lru_cache(maxsize=None, typed=False)
    def typeIsNum(self, col: str) -> bool:
        """Check whether specified column's data type is numerical."""
        return is_num(self.type(col))

    @property
    def possibleFeatureCols(self) -> Set[str]:
        """Possible feature columns for ML modeling."""
        return {col for col in self.contentCols if is_possible_feature(self.type(col))}

    @property
    def possibleCatCols(self) -> Set[str]:
        """Possible categorical content columns."""
        return {col for col in self.contentCols if is_possible_cat(self.type(col))}

    # ====================
    # MAP/REDUCE & related
    # --------------------
    # map
    # reduce
    # __getitem__
    # castType
    # collect

    def map(self,
            mappers: Optional[CallablesType] = None, /,
            reduceMustInclCols: Optional[ColsType] = None,
            **kwargs: Any) -> S3ParquetDataFeeder:
        """Apply mapper function(s) to files."""
        if mappers is None:
            mappers: Tuple[callable] = ()

        if reduceMustInclCols is None:
            reduceMustInclCols: Set[str] = set()

        inheritCache: bool = kwargs.pop('inheritCache', False)
        inheritNRows: bool = kwargs.pop('inheritNRows', inheritCache)

        s3ParquetDF: S3ParquetDataFeeder = \
            S3ParquetDataFeeder(
                path=self.path, awsRegion=self.awsRegion,

                iCol=self._iCol, tCol=self._tCol,
                _mappers=self._mappers + to_iterable(mappers, iterable_type=tuple),
                _reduceMustInclCols=(self._reduceMustInclCols |
                                     to_iterable(reduceMustInclCols, iterable_type=set)),

                reprSampleMinNFiles=self._reprSampleMinNFiles,
                reprSampleSize=self._reprSampleSize,

                minNonNullProportion=self._minNonNullProportion,
                outlierTailProportion=self._outlierTailProportion,
                maxNCats=self._maxNCats,
                minProportionByMaxNCats=self._minProportionByMaxNCats,

                **kwargs)

        if inheritCache:
            s3ParquetDF._inheritCache(self)

        if inheritNRows:
            s3ParquetDF._cache.approxNRows = self._cache.approxNRows
            s3ParquetDF._cache.nRows = self._cache.nRows

        return s3ParquetDF

    def reduce(self, *filePaths: str, **kwargs: Any) -> ReducedDataSetType:
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Reduce from mapped content."""
        _CHUNK_SIZE: int = 10 ** 5

        cols: Optional[Collection[str]] = kwargs.get('cols')
        cols: Set[str] = to_iterable(cols, iterable_type=set) if cols else set()

        nSamplesPerFile: int = kwargs.get('nSamplesPerFile')

        reducer: callable = kwargs.get('reducer',
                                       lambda results:
                                           vstack(tup=results)
                                           if isinstance(results[0], ndarray)
                                           else concat(objs=results,
                                                       axis='index',
                                                       join='outer',
                                                       ignore_index=False,
                                                       keys=None,
                                                       levels=None,
                                                       names=None,
                                                       verify_integrity=False,
                                                       sort=False,
                                                       copy=False))

        verbose: bool = kwargs.pop('verbose', True)

        if not filePaths:
            filePaths: Set[str] = self.filePaths

        results: List[ReducedDataSetType] = []

        # pylint: disable=too-many-nested-blocks
        for filePath in (tqdm(filePaths) if verbose and (len(filePaths) > 1) else filePaths):
            fileLocalPath: Path = self.fileLocalPath(filePath=filePath)

            fileCache: Namespace = self._readMetadataAndSchema(filePath=filePath)

            colsForFile: Set[str] = (
                cols
                if cols
                else fileCache.srcColsInclPartitionKVs
            ) | self._reduceMustInclCols

            srcCols: Set[str] = colsForFile & fileCache.srcColsExclPartitionKVs

            partitionKeyCols: Set[str] = colsForFile.intersection(fileCache.partitionKVs)

            if srcCols:
                pandasDFConstructed: bool = False

                if toSubSample := nSamplesPerFile and (nSamplesPerFile < fileCache.nRows):
                    intermediateN: float = (nSamplesPerFile * fileCache.nRows) ** .5

                    if ((nChunksForIntermediateN := int(math.ceil(intermediateN / _CHUNK_SIZE)))
                            < (approxNChunks := int(math.ceil(fileCache.nRows / _CHUNK_SIZE)))):
                        # arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table
                        fileArrowTable: Table = read_table(source=fileLocalPath,
                                                           columns=list(srcCols),
                                                           use_threads=True,
                                                           metadata=None,
                                                           use_pandas_metadata=True,
                                                           memory_map=False,
                                                           read_dictionary=None,
                                                           filesystem=None,
                                                           filters=None,
                                                           buffer_size=0,
                                                           partitioning='hive',
                                                           use_legacy_dataset=False,
                                                           ignore_prefixes=None,
                                                           pre_buffer=True,
                                                           coerce_int96_timestamp_unit=None)

                        chunkRecordBatches: List[RecordBatch] = \
                            fileArrowTable.to_batches(max_chunksize=_CHUNK_SIZE)

                        nChunks: int = len(chunkRecordBatches)

                        assert nChunks in (approxNChunks - 1, approxNChunks), \
                            ValueError(f'*** {filePath}: {nChunks} vs. '
                                       f'{approxNChunks} Record Batches ***')

                        assert nChunksForIntermediateN <= nChunks, \
                            ValueError(f'*** {filePath}: {nChunksForIntermediateN} vs. '
                                       f'{nChunks} Record Batches ***')

                        chunkPandasDFs: List[DataFrame] = []

                        nSamplesPerChunk: int = int(math.ceil(nSamplesPerFile /
                                                              nChunksForIntermediateN))

                        for chunkRecordBatch in randomSample(population=chunkRecordBatches,
                                                             sampleSize=nChunksForIntermediateN,
                                                             returnCollectionType=tuple):
                            # arrow.apache.org/docs/python/generated/pyarrow.RecordBatch.html
                            # #pyarrow.RecordBatch.to_pandas
                            chunkPandasDF: DataFrame = \
                                chunkRecordBatch.to_pandas(
                                    memory_pool=None,
                                    categories=None,
                                    strings_to_categorical=False,
                                    zero_copy_only=False,

                                    integer_object_nulls=False,
                                    # TODO: check
                                    # (bool, default False) –
                                    # Cast integers with nulls to objects

                                    date_as_object=True,
                                    # TODO: check
                                    # (bool, default True) –
                                    # Cast dates to objects.
                                    # If False, convert to datetime64[ns] dtype.

                                    timestamp_as_object=False,
                                    use_threads=True,

                                    deduplicate_objects=True,
                                    # TODO: check
                                    # (bool, default False) –
                                    # Do not create multiple copies Python objects when created,
                                    # to save on memory use. Conversion will be slower.

                                    ignore_metadata=False,
                                    safe=True,

                                    split_blocks=True,
                                    # TODO: check
                                    # (bool, default False) –
                                    # If True, generate one internal “block”
                                    # for each column when creating a pandas.DataFrame
                                    # from a RecordBatch or Table.
                                    # While this can temporarily reduce memory
                                    # note that various pandas operations can
                                    # trigger “consolidation” which may balloon memory use.

                                    self_destruct=True,
                                    # TODO: check
                                    # EXPERIMENTAL: If True, attempt to deallocate
                                    # the originating Arrow memory while
                                    # converting the Arrow object to pandas.
                                    # If you use the object after calling to_pandas
                                    # with this option it will crash your program.
                                    # Note that you may not see always memory usage improvements.
                                    # For example, if multiple columns share
                                    # an underlying allocation, memory can’t be freed
                                    # until all columns are converted.

                                    types_mapper=None)

                            for k in partitionKeyCols:
                                chunkPandasDF[k] = fileCache.partitionKVs[k]

                            if nSamplesPerChunk < len(chunkPandasDF):
                                chunkPandasDF: DataFrame = \
                                    chunkPandasDF.sample(n=nSamplesPerChunk,
                                                         # frac=None,
                                                         replace=False,
                                                         weights=None,
                                                         random_state=None,
                                                         axis='index',
                                                         ignore_index=False)

                            chunkPandasDFs.append(chunkPandasDF)

                        filePandasDF: DataFrame = concat(objs=chunkPandasDFs,
                                                         axis='index',
                                                         join='outer',
                                                         ignore_index=False,
                                                         keys=None,
                                                         levels=None,
                                                         names=None,
                                                         verify_integrity=False,
                                                         sort=False,
                                                         copy=False)

                        pandasDFConstructed: bool = True

                if not pandasDFConstructed:
                    # pandas.pydata.org/docs/reference/api/pandas.read_parquet
                    filePandasDF: DataFrame = read_parquet(
                        path=fileLocalPath,
                        engine='pyarrow',
                        columns=list(srcCols),
                        storage_options=None,
                        use_nullable_dtypes=True,

                        # arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table:
                        use_threads=True,
                        metadata=None,
                        use_pandas_metadata=True,
                        memory_map=False,
                        read_dictionary=None,
                        filesystem=None,
                        filters=None,
                        buffer_size=0,
                        partitioning='hive',
                        use_legacy_dataset=False,
                        ignore_prefixes=None,
                        pre_buffer=True,
                        coerce_int96_timestamp_unit=None,

                        # arrow.apache.org/docs/python/generated/pyarrow.Table.html
                        # #pyarrow.Table.to_pandas:
                        # memory_pool=None,   # (default)
                        # categories=None,   # (default)
                        # strings_to_categorical=False,   # (default)
                        # zero_copy_only=False,   # (default)

                        # integer_object_nulls=False,   # (default)
                        # TODO: check
                        # (bool, default False) –
                        # Cast integers with nulls to objects

                        # date_as_object=True,   # (default)
                        # TODO: check
                        # (bool, default True) –
                        # Cast dates to objects.
                        # If False, convert to datetime64[ns] dtype.

                        # timestamp_as_object=False,   # (default)
                        # use_threads=True,   # (default)

                        # deduplicate_objects=True,   # (default: *** False ***)
                        # TODO: check
                        # (bool, default False) –
                        # Do not create multiple copies Python objects when created,
                        # to save on memory use. Conversion will be slower.

                        # ignore_metadata=False,   # (default)
                        # safe=True,   # (default)

                        # split_blocks=True,   # (default: *** False ***)
                        # TODO: check
                        # (bool, default False) –
                        # If True, generate one internal “block” for each column
                        # when creating a pandas.DataFrame from a RecordBatch or Table.
                        # While this can temporarily reduce memory note that
                        # various pandas operations can trigger “consolidation”
                        # which may balloon memory use.

                        # self_destruct=True,   # (default: *** False ***)
                        # TODO: check
                        # EXPERIMENTAL: If True, attempt to deallocate the originating
                        # Arrow memory while converting the Arrow object to pandas.
                        # If you use the object after calling to_pandas with this option
                        # it will crash your program.
                        # Note that you may not see always memory usage improvements.
                        # For example, if multiple columns share an underlying allocation,
                        # memory can’t be freed until all columns are converted.

                        # types_mapper=None,   # (default)
                    )

                    for k in partitionKeyCols:
                        filePandasDF[k] = fileCache.partitionKVs[k]

                    if toSubSample:
                        filePandasDF: DataFrame = filePandasDF.sample(n=nSamplesPerFile,
                                                                      # frac=None,
                                                                      replace=False,
                                                                      weights=None,
                                                                      random_state=None,
                                                                      axis='index',
                                                                      ignore_index=False)

            else:
                filePandasDF: DataFrame = DataFrame(index=range(nSamplesPerFile
                                                                if nSamplesPerFile and
                                                                (nSamplesPerFile < fileCache.nRows)
                                                                else fileCache.nRows))

                for k in partitionKeyCols:
                    filePandasDF[k] = fileCache.partitionKVs[k]

            result: ReducedDataSetType = filePandasDF
            for mapper in self._mappers:
                result: ReducedDataSetType = mapper(result)

            results.append(result)

        return reducer(results)

    @staticmethod
    def _getCols(pandasDF: DataFrame, cols: Union[str, Tuple[str]]) -> DataFrame:
        for missingCol in to_iterable(cols, iterable_type=set).difference(pandasDF.columns):
            pandasDF.loc[:, missingCol] = None

        return pandasDF[cols if isinstance(cols, str) else list(cols)]

    @lru_cache(maxsize=None, typed=False)
    def __getitem__(self, cols: Union[str, Tuple[str]], /) -> S3ParquetDataFeeder:
        """Get column(s)."""
        return self.map(partial(self._getCols, cols=cols),
                        reduceMustInclCols=cols,
                        inheritNRows=True)

    @lru_cache(maxsize=None, typed=False)
    def castType(self, **colsToTypes: Dict[str, Any]) -> S3ParquetDataFeeder:
        """Cast data type(s) of column(s)."""
        return self.map(lambda df: df.astype(colsToTypes, copy=False, errors='raise'),
                        reduceMustInclCols=set(colsToTypes),
                        inheritNRows=True)

    def collect(self, *cols: str, **kwargs: Any) -> ReducedDataSetType:
        """Collect content."""
        return self.reduce(cols=cols if cols else None, **kwargs)

    # =========================
    # KEY (SETTABLE) PROPERTIES
    # -------------------------
    # iCol
    # tCol

    @property
    def iCol(self) -> Optional[str]:
        """Entity/Identity column."""
        return self._iCol

    @iCol.setter
    def iCol(self, iCol: str):
        if iCol != self._iCol:
            self._iCol: Optional[str] = iCol

            if iCol is not None:
                assert iCol, ValueError(f'*** iCol {iCol} INVALID ***')

    @iCol.deleter
    def iCol(self):
        self._iCol: Optional[str] = None

    @property
    def tCol(self) -> Optional[str]:
        """Date-Time column."""
        return self._tCol

    @tCol.setter
    def tCol(self, tCol: str):
        if tCol != self._tCol:
            self._tCol: Optional[str] = tCol

            if tCol is not None:
                assert tCol, ValueError(f'*** tCol {tCol} INVALID ***')

    @tCol.deleter
    def tCol(self):
        self._tCol: Optional[str] = None

    # ===========
    # REPR SAMPLE
    # -----------
    # prelimReprSamplePiecePaths
    # reprSamplePiecePaths
    # _assignReprSample

    @property
    def prelimReprSamplePiecePaths(self) -> Set[str]:
        """Prelim Representative Sample Piece Paths."""
        if self._cache.prelimReprSamplePiecePaths is None:
            self._cache.prelimReprSamplePiecePaths = \
                randomSample(population=self.piecePaths,
                             sampleSize=self._reprSampleMinNPieces)

        return self._cache.prelimReprSamplePiecePaths

    @property
    def reprSamplePiecePaths(self) -> Set[str]:
        """Return representative sample piece paths."""
        if self._cache.reprSamplePiecePaths is None:
            reprSampleNPieces: int = \
                int(math.ceil(
                    ((min(self._reprSampleSize, self.approxNRows) / self.approxNRows) ** .5)
                    * self.nPieces))

            self._cache.reprSamplePiecePaths = (
                self._cache.prelimReprSamplePiecePaths |
                (randomSample(
                    population=self.piecePaths - self._cache.prelimReprSamplePiecePaths,
                    sampleSize=reprSampleNPieces - self._reprSampleMinNPieces)
                 if reprSampleNPieces > self._reprSampleMinNPieces
                 else set()))

        return self._cache.reprSamplePiecePaths

    def _assignReprSample(self):
        self._cache.reprSample = self.sample(n=self._reprSampleSize,
                                             piecePaths=self.reprSamplePiecePaths,
                                             verbose=True)

        # pylint: disable=attribute-defined-outside-init
        self._reprSampleSize: int = len(self._cache.reprSample)

        self._cache.nonNullProportion = {}
        self._cache.suffNonNull = {}

    # ==============
    # SUBSET METHODS
    # --------------
    # _subset
    # filterByPartitionKeys
    # sample

    @lru_cache(maxsize=None, typed=False)   # computationally expensive, so cached
    def _subset(self, *piecePaths: str, **kwargs: Any) -> S3ParquetDataFeeder:
        # pylint: disable=too-many-locals
        if piecePaths:
            assert self.piecePaths.issuperset(piecePaths)

            nPiecePaths: int = len(piecePaths)

            if nPiecePaths == self.nPieces:
                return self

            if nPiecePaths > 1:
                verbose: bool = kwargs.pop('verbose', True)

                subsetDirS3Key: str = f'{self.tmpDirS3Key}/{uuid4()}'

                _pathPlusSepLen: int = len(self.path) + 1

                subsetPath: str = f's3://{self.s3Bucket}/{subsetDirS3Key}'

                if verbose:
                    self.stdOutLogger.info(
                        msg=(msg := f'Subsetting {len(piecePaths):,} Pieces to "{subsetPath}"...'))
                    tic: float = time.time()

                for piecePath in (tqdm(piecePaths) if verbose else piecePaths):
                    pieceSubPath: str = piecePath[_pathPlusSepLen:]

                    _from_key: str = f'{self.pathS3Key}/{pieceSubPath}'
                    _to_key: str = f'{subsetDirS3Key}/{pieceSubPath}'

                    try:
                        self.S3_CLIENT.copy(
                            CopySource=dict(Bucket=self.s3Bucket,
                                            Key=_from_key),
                            Bucket=self.s3Bucket,
                            Key=_to_key)

                    except Exception as err:
                        print(f'*** FAILED TO COPY FROM "{_from_key}" '
                              f'TO "{_to_key}" ***')

                        raise err

                if verbose:
                    toc: float = time.time()
                    self.stdOutLogger.info(msg=f'{msg} done!   <{toc-tic:.1f}>')

            else:
                subsetPath: str = piecePaths[0]

            return S3ParquetDataFeeder(
                path=subsetPath, awsRegion=self.awsRegion,

                iCol=self._iCol, tCol=self._tCol,
                _mappers=self._mappers,
                _reduceMustInclCols=self._reduceMustInclCols,

                reprSampleMinNPieces=self._reprSampleMinNPieces,
                reprSampleSize=self._reprSampleSize,

                minNonNullProportion=self._minNonNullProportion,
                outlierTailProportion=self._outlierTailProportion,
                maxNCats=self._maxNCats,
                minProportionByMaxNCats=self._minProportionByMaxNCats,

                **kwargs)

        return self

    @lru_cache(maxsize=None, typed=False)
    def filterByPartitionKeys(self,
                              *filterCriteriaTuples: Union[Tuple[str, str],
                                                           Tuple[str, str, str]],
                              **kwargs: Any) -> S3ParquetDataFeeder:
        # pylint: disable=too-many-branches
        """Filter by partition keys."""
        filterCriteria: Dict[str, Tuple[Optional[str], Optional[str], Optional[Set[str]]]] = {}

        _samplePiecePath: str = next(iter(self.piecePaths))

        for filterCriteriaTuple in filterCriteriaTuples:
            assert isinstance(filterCriteriaTuple, PY_LIST_OR_TUPLE)
            filterCriteriaTupleLen = len(filterCriteriaTuple)

            col: str = filterCriteriaTuple[0]

            if f'{col}=' in _samplePiecePath:
                if filterCriteriaTupleLen == 2:
                    fromVal: Optional[str] = None
                    toVal: Optional[str] = None
                    inSet: Set[str] = {str(v) for v in to_iterable(filterCriteriaTuple[1])}

                elif filterCriteriaTupleLen == 3:
                    fromVal: Optional[str] = filterCriteriaTuple[1]
                    if fromVal is not None:
                        fromVal: str = str(fromVal)

                    toVal: Optional[str] = filterCriteriaTuple[2]
                    if toVal is not None:
                        toVal: str = str(toVal)

                    inSet: Optional[Set[str]] = None

                else:
                    raise ValueError(
                        f'*** {type(self)} FILTER CRITERIA MUST BE EITHER '
                        '(<colName>, <fromVal>, <toVal>) OR '
                        '(<colName>, <inValsSet>) ***')

                filterCriteria[col] = fromVal, toVal, inSet

        if filterCriteria:
            piecePaths: Set[str] = set()

            for piecePath in self.piecePaths:
                pieceSatisfiesCriteria: bool = True

                for col, (fromVal, toVal, inSet) in filterCriteria.items():
                    v: str = re.search(f'{col}=(.*?)/', piecePath).group(1)

                    # pylint: disable=too-many-boolean-expressions
                    if ((fromVal is not None) and (v < fromVal)) or \
                            ((toVal is not None) and (v > toVal)) or \
                            ((inSet is not None) and (v not in inSet)):
                        pieceSatisfiesCriteria: bool = False
                        break

                if pieceSatisfiesCriteria:
                    piecePaths.add(piecePath)

            assert piecePaths, FileNotFoundError(f'*** {self}: NO PIECE PATHS SATISFYING '
                                                 f'FILTER CRITERIA {filterCriteria} ***')

            return self._subset(*piecePaths, **kwargs)

        return self

    def sample(self, *cols: str, **kwargs: Any) -> ReducedDataSetType:
        """Sample."""
        n: int = kwargs.pop('n', self._DEFAULT_REPR_SAMPLE_SIZE)

        piecePaths: Optional[Collection[str]] = kwargs.pop('piecePaths', None)

        verbose: bool = kwargs.pop('verbose', True)

        if piecePaths:
            nSamplePieces: int = len(piecePaths)

        else:
            minNPieces: int = kwargs.pop('minNPieces', self._reprSampleMinNPieces)
            maxNPieces: Optional[int] = kwargs.pop('maxNPieces', None)

            nSamplePieces: int = (
                max(int(math.ceil(
                        ((min(n, self.approxNRows) / self.approxNRows) ** .5)
                        * self.nPieces)),
                    minNPieces)
                if (self.nPieces > 1) and ((maxNPieces is None) or (maxNPieces > 1))
                else 1)

            if maxNPieces:
                nSamplePieces: int = min(nSamplePieces, maxNPieces)

            if nSamplePieces < self.nPieces:
                piecePaths: Set[str] = randomSample(population=self.piecePaths,
                                                    sampleSize=nSamplePieces)
            else:
                nSamplePieces: int = self.nPieces
                piecePaths: Set[str] = self.piecePaths

        if verbose or debug.ON:
            self.stdOutLogger.info(
                msg=f"Sampling {n:,} Rows{f' of Columns {cols}' if cols else ''} "
                    f'from {nSamplePieces:,} Pieces...')

        return self.reduce(*piecePaths,
                           cols=cols,
                           nSamplesPerPiece=int(math.ceil(n / nSamplePieces)),
                           verbose=verbose,
                           **kwargs)

    # ================
    # COLUMN PROFILING
    # ----------------
    # count
    # nonNullProportion
    # distinct
    # quantile
    # sampleStat
    # outlierRstStat / outlierRstMin / outlierRstMax
    # profile

    def count(self, *cols: str, **kwargs: Any) -> Union[int, Namespace]:
        """Count non-NULL values in specified column(s).

        Return:
            - If 1 column name is given,
            return its corresponding non-``NULL`` count

            - If multiple column names are given,
            return a {``col``: corresponding non-``NULL`` count} *dict*

            - If no column names are given,
            return a {``col``: corresponding non-``NULL`` count} *dict*
            for all columns
        """
        if not cols:
            cols: Set[str] = self.contentCols

        if len(cols) > 1:
            return Namespace(**{col: self.count(col, **kwargs) for col in cols})

        col: str = cols[0]

        pandasDF: Optional[DataFrame] = kwargs.get('pandasDF')

        lowerNumericNull, upperNumericNull = self._nulls[col]

        if pandasDF is None:
            if col not in self._cache.count:
                verbose: Optional[bool] = True if debug.ON else kwargs.get('verbose')

                if verbose:
                    tic: float = time.time()

                self._cache.count[col] = result = int(
                    self[col]
                    .map(((lambda series: (series.notnull()
                                           .sum(axis='index',
                                                skipna=True,
                                                level=None,
                                                # numeric_only=True,
                                                min_count=0)))

                          if isnull(upperNumericNull)

                          else (lambda series: ((series < upperNumericNull)
                                                .sum(axis='index',
                                                     skipna=True,
                                                     level=None,
                                                     # numeric_only=True,
                                                     min_count=0))))

                         if isnull(lowerNumericNull)

                         else ((lambda series: ((series > lowerNumericNull)
                                                .sum(axis='index',
                                                     skipna=True,
                                                     level=None,
                                                     # numeric_only=True,
                                                     min_count=0)))

                               if isnull(upperNumericNull)

                               else (lambda series: (series
                                                     .between(left=lowerNumericNull,
                                                              right=upperNumericNull,
                                                              inclusive='neither')
                                                     .sum(axis='index',
                                                          skipna=True,
                                                          level=None,
                                                          # numeric_only=True,
                                                          min_count=0)))),
                         reduceMustInclCols=col)
                    .reduce(cols=col, reducer=sum))

                if verbose:
                    toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'No. of Non-NULLs of Column "{col}" = '
                            f'{result:,}   <{toc - tic:,.1f} s>')

            return self._cache.count[col]

        series: Series = pandasDF[col]

        series: Series = ((series.notnull()

                           if isnull(upperNumericNull)

                           else (series < upperNumericNull))

                          if isnull(lowerNumericNull)

                          else ((series > lowerNumericNull)

                                if isnull(upperNumericNull)

                                else series.between(left=lowerNumericNull,
                                                    right=upperNumericNull,
                                                    inclusive='neither')))

        return series.sum(axis='index',
                          skipna=True,
                          level=None,
                          # numeric_only=True,
                          min_count=0)

    def nonNullProportion(self, *cols: str, **kwargs: Any) -> Union[float, Namespace]:
        """Calculate non-NULL data proportion(s) of specified column(s).

        Return:
            - If 1 column name is given,
            return its *approximate* non-``NULL`` proportion

            - If multiple column names are given,
            return {``col``: approximate non-``NULL`` proportion} *dict*

            - If no column names are given,
            return {``col``: approximate non-``NULL`` proportion}
            *dict* for all columns
        """
        if not cols:
            cols: Set[str] = self.contentCols

        if len(cols) > 1:
            return Namespace(**{col: self.nonNullProportion(col, **kwargs) for col in cols})

        col: str = cols[0]

        if col not in self._cache.nonNullProportion:
            self._cache.nonNullProportion[col] = (
                self.count(col, pandasDF=self.reprSample, **kwargs) /
                self.reprSampleSize)

        return self._cache.nonNullProportion[col]

    def distinct(self, *cols: str, **kwargs: Any) -> Union[Series, Namespace]:
        """Return distinct values in specified column(s).

        Return:
            *Approximate* list of distinct values of ``ADF``'s column ``col``,
                with optional descending-sorted counts for those values

        Args:
            col (str): name of a column

            count (bool): whether to count the number of appearances
            of each distinct value of the specified ``col``
        """
        if not cols:
            cols: Set[str] = self.contentCols

        asDict: bool = kwargs.pop('asDict', False)

        if len(cols) > 1:
            return Namespace(**{col: self.distinct(col, **kwargs) for col in cols})

        col: str = cols[0]

        if col not in self._cache.distinct:
            self._cache.distinct[col] = \
                self.reprSample[col].value_counts(normalize=True,
                                                  sort=True,
                                                  ascending=False,
                                                  bins=None,
                                                  dropna=False)

        return (Namespace(**{col: self._cache.distinct[col]})
                if asDict
                else self._cache.distinct[col])

    @lru_cache(maxsize=None, typed=False)   # computationally expensive, so cached
    def quantile(self, *cols: str, **kwargs: Any) -> Union[float, int,
                                                           Series, Namespace]:
        """Return quantile values in specified column(s)."""
        if len(cols) > 1:
            return Namespace(**{col: self.quantile(col, **kwargs) for col in cols})

        col: str = cols[0]

        # for precision, calc from whole data set instead of from reprSample
        return self[col].reduce(cols=col).quantile(q=kwargs.get('q', .5),
                                                   interpolation='linear')

    def sampleStat(self, *cols: str, **kwargs: Any) -> Union[float, int, Namespace]:
        """Approximate measurements of a certain stat on numerical columns.

        Args:
            *cols (str): column name(s)
            **kwargs:
                - **stat**: one of the following:
                    - ``avg``/``mean`` (default)
                    - ``median``
                    - ``min``
                    - ``max``
        """
        if not cols:
            cols: Set[str] = self.possibleNumCols

        if len(cols) > 1:
            return Namespace(**{col: self.sampleStat(col, **kwargs) for col in cols})

        col: str = cols[0]

        if self.typeIsNum(col):
            stat: str = kwargs.pop('stat', 'mean').lower()
            if stat == 'avg':
                stat: str = 'mean'
            capitalizedStatName: str = stat.capitalize()
            s: str = f'sample{capitalizedStatName}'

            if s not in self._cache:
                setattr(self._cache, s, {})
            _cache: Dict[str, PyNumType] = getattr(self._cache, s)

            if col not in _cache:
                verbose: Optional[bool] = True if debug.ON else kwargs.get('verbose')

                if verbose:
                    tic: float = time.time()

                result = getattr(self.reprSample[col], stat)(axis='index', skipna=True, level=None)

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result: float = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result: int = int(result)

                assert isinstance(result, PY_NUM_TYPES + (NAType,)), \
                    TypeError(f'*** "{col}" SAMPLE '
                              f'{capitalizedStatName.upper()} = '
                              f'{result} ({type(result)}) ***')

                if verbose:
                    toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'Sample {capitalizedStatName} for '
                            f'Column "{col}" = '
                            f'{result:,.3g}   <{toc - tic:,.1f} s>')

                _cache[col] = result

            return _cache[col]

        raise ValueError(f'*** {self}.sampleStat({col}, ...): '
                         f'COLUMN "{col}" NOT NUMERICAL ***')

    def outlierRstStat(self, *cols: str, **kwargs: Any) -> Union[float, int, Namespace]:
        # pylint: disable=too-many-branches
        """Return outlier-resistant stat for specified column(s)."""
        if not cols:
            cols: Set[str] = self.possibleNumCols

        if len(cols) > 1:
            return Namespace(**{col: self.outlierRstStat(col, **kwargs) for col in cols})

        col: str = cols[0]

        if self.typeIsNum(col):
            stat: str = kwargs.pop('stat', 'mean').lower()
            if stat == 'avg':
                stat: str = 'mean'
            capitalizedStatName: str = stat.capitalize()
            s: str = f'outlierRst{capitalizedStatName}'

            if hasattr(self, s):
                return getattr(self, s)(col, **kwargs)

            if s not in self._cache:
                setattr(self._cache, s, {})
            _cache: Dict[str, PyNumType] = getattr(self._cache, s)

            if col not in _cache:
                verbose: Optional[bool] = True if debug.ON else kwargs.get('verbose')

                if verbose:
                    tic: float = time.time()

                series: Series = self.reprSample[col]

                outlierTails: Optional[str] = kwargs.pop('outlierTails', 'both')

                if outlierTails == 'both':
                    series: Series = series.loc[
                        series.between(left=self.outlierRstMin(col),
                                       right=self.outlierRstMax(col),
                                       inclusive='both')]

                elif outlierTails == 'lower':
                    series: Series = series.loc[series >= self.outlierRstMin(col)]

                elif outlierTails == 'upper':
                    series: Series = series.loc[series <= self.outlierRstMax(col)]

                result = getattr(series, stat)(axis='index', skipna=True, level=None)

                if isnull(result):
                    self.stdOutLogger.warning(
                        msg=f'*** "{col}" OUTLIER-RESISTANT '
                            f'{capitalizedStatName.upper()} = '
                            f'{result} ***')

                    result: PyNumType = self.outlierRstMin(col)

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result: float = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result: int = int(result)

                assert isinstance(result, PY_NUM_TYPES + (NAType,)), \
                    TypeError(f'*** "{col}" '
                              f'OUTLIER-RESISTANT {capitalizedStatName.upper()}'
                              f' = {result} ({type(result)}) ***')

                if verbose:
                    toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'Outlier-Resistant {capitalizedStatName}'
                            f' for Column "{col}" = '
                            f'{result:,.3g}   <{toc - tic:,.1f} s>')

                _cache[col] = result

            return _cache[col]

        raise ValueError(f'*** {self}.outlierRstStat({col}, ...): '
                         f'COLUMN "{col}" NOT NUMERICAL ***')

    def outlierRstMin(self, *cols: str, **kwargs: Any) -> Union[float, int, Namespace]:
        """Return outlier-resistant minimum for specified column(s)."""
        if not cols:
            cols: Set[str] = self.possibleNumCols

        if len(cols) > 1:
            return Namespace(**{col: self.outlierRstMin(col, **kwargs) for col in cols})

        col: str = cols[0]

        if self.typeIsNum(col):
            if 'outlierRstMin' not in self._cache:
                self._cache.outlierRstMin = {}

            if col not in self._cache.outlierRstMin:
                verbose: Optional[bool] = True if debug.ON else kwargs.get('verbose')

                if verbose:
                    tic: float = time.time()

                series: Series = self.reprSample[col]

                outlierRstMin: PyNumType = \
                    series.quantile(q=self._outlierTailProportion[col],
                                    interpolation='linear')

                sampleMin: PyNumType = self.sampleStat(col, stat='min')
                sampleMedian: PyNumType = self.sampleStat(col, stat='median')

                result = (series.loc[series > sampleMin].min(axis='index', skipna=True, level=None)
                          if (outlierRstMin == sampleMin) and (outlierRstMin < sampleMedian)
                          else outlierRstMin)

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result: float = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result: int = int(result)

                assert isinstance(result, PY_NUM_TYPES + (NAType,)), \
                    TypeError(f'*** "{col}" OUTLIER-RESISTANT MIN = '
                              f'{result} ({type(result)}) ***')

                if verbose:
                    toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'Outlier-Resistant Min of Column "{col}" = '
                            f'{result:,.3g}   <{toc - tic:,.1f} s>')

                self._cache.outlierRstMin[col] = result

            return self._cache.outlierRstMin[col]

        raise ValueError(f'*** {self}.outlierRstMin({col}, ...): '
                         f'COLUMN "{col}" NOT NUMERICAL ***')

    def outlierRstMax(self, *cols: str, **kwargs: Any) -> Union[float, int, Namespace]:
        """Return outlier-resistant maximum for specified column(s)."""
        if not cols:
            cols: Set[str] = self.possibleNumCols

        if len(cols) > 1:
            return Namespace(**{col: self.outlierRstMax(col, **kwargs) for col in cols})

        col: str = cols[0]

        if self.typeIsNum(col):
            if 'outlierRstMax' not in self._cache:
                self._cache.outlierRstMax = {}

            if col not in self._cache.outlierRstMax:
                verbose: Optional[bool] = True if debug.ON else kwargs.get('verbose')

                if verbose:
                    tic: float = time.time()

                series: Series = self.reprSample[col]

                outlierRstMax: PyNumType = \
                    series.quantile(q=1 - self._outlierTailProportion[col],
                                    interpolation='linear')

                sampleMax: PyNumType = self.sampleStat(col, stat='max')
                sampleMedian: PyNumType = self.sampleStat(col, stat='median')

                result = (series.loc[series < sampleMax].max(axis='index', skipna=True, level=None)
                          if (outlierRstMax == sampleMax) and (outlierRstMax > sampleMedian)
                          else outlierRstMax)

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result: float = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result: int = int(result)

                assert isinstance(result, PY_NUM_TYPES + (NAType,)), \
                    TypeError(f'*** "{col}" OUTLIER-RESISTANT MAX = {result} '
                              f'({type(result)}) ***')

                if verbose:
                    toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'Outlier-Resistant Max of Column "{col}" = '
                            f'{result:,.3g}   <{toc - tic:,.1f} s>')

                self._cache.outlierRstMax[col] = result

            return self._cache.outlierRstMax[col]

        raise ValueError(f'*** {self}.outlierRstMax({col}, ...): '
                         f'COLUMN "{col}" NOT NUMERICAL ***')

    def profile(self, *cols: str, **kwargs: Any) -> Namespace:
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Profile specified column(s).

        Return:
            *dict* of profile of salient statistics on
            specified columns of ``ADF``

        Args:
            *cols (str): names of column(s) to profile

            **kwargs:

                - **profileCat** *(bool, default = True)*:
                whether to profile possible categorical columns

                - **profileNum** *(bool, default = True)*:
                whether to profile numerical columns

                - **skipIfInsuffNonNull** *(bool, default = False)*:
                whether to skip profiling if column does not have
                enough non-NULLs
        """
        if not cols:
            cols: Set[str] = self.contentCols

        asDict: bool = kwargs.pop('asDict', False)

        if len(cols) > 1:
            return Namespace(**{col: self.profile(col, **kwargs) for col in cols})

        col: str = cols[0]

        verbose: Optional[Union[bool, int]] = True if debug.ON else kwargs.get('verbose')

        if verbose:
            self.stdOutLogger.info(msg=(msg := f'Profiling Column "{col}"...'))
            tic: float = time.time()

        colType: DataType = self.type(col)
        profile: Namespace = Namespace(type=colType)

        # non-NULL Proportions
        profile.nonNullProportion = self.nonNullProportion(col, verbose=verbose > 1)

        if self.suffNonNull(col) or (not kwargs.get('skipIfInsuffNonNull', False)):
            # profile categorical column
            if kwargs.get('profileCat', True) and is_possible_cat(colType):
                profile.distinctProportions = self.distinct(col, verbose=verbose > 1)

            # profile numerical column
            if kwargs.get('profileNum', True) and self.typeIsNum(col):
                outlierTailProportion: float = self._outlierTailProportion[col]

                quantilesOfInterest: Series = Series(index=(0,
                                                            outlierTailProportion,
                                                            .5,
                                                            1 - outlierTailProportion,
                                                            1))
                quantileProbsToQuery: List[float] = []

                sampleMin: Optional[PyNumType] = self._cache.sampleMin.get(col)
                if calcAndCacheSampleMin := (sampleMin is None):
                    quantileProbsToQuery.append(0.)
                else:
                    quantilesOfInterest[0] = sampleMin

                outlierRstMin: Optional[PyNumType] = self._cache.outlierRstMin.get(col)
                if calcAndCacheOutlierRstMin := (outlierRstMin is None):
                    quantileProbsToQuery.append(outlierTailProportion)
                else:
                    quantilesOfInterest[outlierTailProportion] = outlierRstMin

                sampleMedian: Optional[PyNumType] = self._cache.sampleMedian.get(col)
                if calcAndCacheSampleMedian := (sampleMedian is None):
                    quantileProbsToQuery.append(.5)
                else:
                    quantilesOfInterest[.5] = sampleMedian

                outlierRstMax: Optional[PyNumType] = self._cache.outlierRstMax.get(col)
                if calcAndCacheOutlierRstMax := (outlierRstMax is None):
                    quantileProbsToQuery.append(1 - outlierTailProportion)
                else:
                    quantilesOfInterest[1 - outlierTailProportion] = outlierRstMax

                sampleMax: Optional[PyNumType] = self._cache.sampleMax.get(col)
                if calcAndCacheSampleMax := (sampleMax is None):
                    quantileProbsToQuery.append(1.)
                else:
                    quantilesOfInterest[1] = sampleMax

                series: Series = self.reprSample[col]

                if quantileProbsToQuery:
                    quantilesOfInterest.mask(
                        cond=quantilesOfInterest.isnull(),
                        other=series.quantile(q=quantileProbsToQuery, interpolation='linear'),
                        inplace=True,
                        axis=None,
                        level=None,
                        errors='raise')

                (sampleMin, outlierRstMin,
                 sampleMedian,
                 outlierRstMax, sampleMax) = quantilesOfInterest

                if calcAndCacheSampleMin:
                    self._cache.sampleMin[col] = sampleMin

                if calcAndCacheOutlierRstMin:
                    if (outlierRstMin == sampleMin) and (outlierRstMin < sampleMedian):
                        outlierRstMin: PyNumType = (
                            series.loc[series > sampleMin]
                            .min(axis='index', skipna=True, level=None))

                    self._cache.outlierRstMin[col] = outlierRstMin

                if calcAndCacheSampleMedian:
                    self._cache.sampleMedian[col] = sampleMedian

                if calcAndCacheOutlierRstMax:
                    if (outlierRstMax == sampleMax) and (outlierRstMax > sampleMedian):
                        outlierRstMax: PyNumType = (
                            series.loc[series < sampleMax]
                            .max(axis='index', skipna=True, level=None))

                    self._cache.outlierRstMax[col] = outlierRstMax

                if calcAndCacheSampleMax:
                    self._cache.sampleMax[col] = sampleMax

                profile.sampleRange = sampleMin, sampleMax
                profile.outlierRstRange = outlierRstMin, outlierRstMax

                profile.sampleMean = self.sampleStat(col, stat='mean', verbose=verbose)

                profile.outlierRstMean = \
                    self._cache.outlierRstMean.get(
                        col,
                        self.outlierRstStat(col, stat='mean', verbose=verbose))

                profile.outlierRstMedian = \
                    self._cache.outlierRstMedian.get(
                        col,
                        self.outlierRstStat(col, stat='median', verbose=verbose))

        if verbose:
            toc: float = time.time()
            self.stdOutLogger.info(msg=f'{msg} done!   <{toc - tic:,.1f} s>')

        return Namespace(**{col: profile}) if asDict else profile

    # =========
    # DATA PREP
    # ---------
    # preprocessForML

        # pylint: disable=too-many-branches,too-many-locals
        """Fill ``NULL``/``NaN`` values for numerical columns.

        Return:
            ``S3ParquetDataFeeder`` with ``NULL``/``NaN`` values filled.

        Args:
            *args (str): names of column(s) to fill/interpolate

            **kwargs:
                - **method** *(str)*: one of the following methods to fill
                    ``NULL`` values in **numerical** columns,
                    or *dict* of such method specifications by column name

                    - ``avg``/``mean`` (default)
                    - ``min``
                    - ``max``
                    - ``None`` (do nothing)

                - **value**: single value, or *dict* of values by column name,
                    to use if ``method`` is ``None`` or not applicable

                - **outlierTails** *(str or dict of str, default = 'both')*:
                specification of in which distribution tail
                (``None``, ``lower``, ``upper`` and ``both`` (default))
                of each numerical column out-lying values may exist
        """
        returnDetails: bool = kwargs.pop('returnDetails', False)

        verbose: Union[bool, int] = kwargs.pop('verbose', False)
        if debug.ON:
            verbose: bool = True

        value: Optional[Union[PyNumType, Dict[str, PyNumType]]] = kwargs.pop('value', None)

        method: Optional[Union[str, Dict[str, Optional[str]]]] = \
            kwargs.pop('method', 'mean' if value is None else None)

        cols: Set[str] = set(cols)

        if isinstance(method, dict):
            cols.update(method)

        if isinstance(value, dict):
            cols.update(value)

        cols &= self.possibleNumCols
        cols -= self.indexCols

        if not cols:
            return self.copy()

        nulls: Dict[str, Tuple[Optional[PyNumType], Optional[PyNumType]]] = kwargs.pop('nulls', {})

        for col in cols:
            if col in nulls:
                colNulls: Tuple[Optional[PyNumType], Optional[PyNumType]] = nulls[col]

                assert (isinstance(colNulls, PY_LIST_OR_TUPLE) and
                        (len(colNulls) == 2) and
                        ((colNulls[0] is None) or isinstance(colNulls[0], PY_NUM_TYPES)) and
                        ((colNulls[1] is None) or isinstance(colNulls[1], PY_NUM_TYPES)))

            else:
                nulls[col] = None, None

        outlierTails: Optional[Union[str, Dict[str, Optional[str]]]] = \
            kwargs.pop('outlierTails', {})
        if not isinstance(outlierTails, DICT_OR_NAMESPACE_TYPES):
            outlierTails: Dict[str, Optional[str]] = {col: outlierTails for col in cols}

        nullFillDetails: Namespace = Namespace()

        if verbose:
            self.stdOutLogger.info(msg = (msg := ('NULL-Filling Columns ' +
                                                  ', '.join(f'"{col}"' for col in cols) +
                                                  '...')))
            tic: float = time.time()

        for col in cols:
            colOutlierTails: Optional[str] = outlierTails.get(col, 'both')

            methodForCol: Optional[str] = (method[col]
                                           if isinstance(method, dict) and (col in method)
                                           else method)

            if methodForCol:
                colFallBackVal: PyNumType = self.outlierRstStat(col,
                                                                stat=methodForCol,
                                                                outlierTails=colOutlierTails,
                                                                verbose=verbose > 1)

            elif isinstance(value, dict):
                colFallBackVal: Optional[PyNumType] = value.get(col)

                if not isinstance(colFallBackVal, PY_NUM_TYPES):
                    colFallBackVal: Optional[PyNumType] = None

            elif isinstance(value, PY_NUM_TYPES):
                colFallBackVal: PyNumType = value

            else:
                colFallBackVal: Optional[PyNumType] = None

            nullFillDetails[col] = (self._NULL_FILL_PREFIX + col,
                                    {'nulls': nulls[col],
                                     'null-fill-method': methodForCol,
                                     'null-fill-value': colFallBackVal})

        s3ParquetDF: S3ParquetDataFeeder = \
            self.map(PandasNumericalNullFiller(nullFillDetails=nullFillDetails),
                     inheritNRows=True, **kwargs)
        s3ParquetDF._inheritCache(self, *cols)
        s3ParquetDF._cache.reprSample = self._cache.reprSample

        if verbose:
            toc: float = time.time()
            self.stdOutLogger.info(msg=f'{msg} done!   <{(toc - tic) / 60:,.1f} m>')

        return (s3ParquetDF, nullFillDetails) if returnDetails else s3ParquetDF

        nulls: Dict[str, Tuple[Optional[PyNumType], Optional[PyNumType]]] = kwargs.pop('nulls', {})

        forceCatIncl: Optional[ColsType] = kwargs.pop('forceCatIncl', None)
        forceCatExcl: Optional[ColsType] = kwargs.pop('forceCatExcl', None)
        forceCat: Optional[ColsType] = kwargs.pop('forceCat', None)
        forceCat: Set[str] = (((set()
                                if forceCat is None
                                else to_iterable(forceCat, iterable_type=set))
                               | (set()
                                  if forceCatIncl is None
                                  else to_iterable(forceCatIncl, iterable_type=set)))
                              - (set()
                                 if forceCatExcl is None
                                 else to_iterable(forceCatExcl, iterable_type=set)))

        scaleCat: bool = kwargs.pop('scaleCat', True)

        forceNumIncl: Optional[ColsType] = kwargs.pop('forceNumIncl', None)
        forceNumExcl: Optional[ColsType] = kwargs.pop('forceNumExcl', None)
        forceNum: Optional[ColsType] = kwargs.pop('forceNum', None)
        forceNum: Set[str] = (((set()
                                if forceNum is None
                                else to_iterable(forceNum, iterable_type=set))
                               | (set()
                                  if forceNumIncl is None
                                  else to_iterable(forceNumIncl, iterable_type=set)))
                              - (set()
                                 if forceNumExcl is None
                                 else to_iterable(forceNumExcl, iterable_type=set)))

        fill: Dict[str, Optional[Union[str, Optional[PyNumType]]]] = \
            kwargs.pop('fill', dict(method='mean', value=None, outlierTails='both'))

        assert fill, ValueError(f'*** {type(self)}.preprocessForML(...) MUST INVOLVE NULL-FILLING '
                                f'FOR NUMERICAL COLS ***')

        scaler: Optional[str] = kwargs.pop('scaler', 'standard')
        if scaler:
            scaler: str = scaler.lower()

        returnNumPy: bool = kwargs.pop('returnNumPy', False)
        returnOrigToPrepColMaps: bool = kwargs.pop('returnOrigToPrepColMaps', False)

        loadPath: Optional[PathType] = kwargs.pop('loadPath', None)
        savePath: Optional[PathType] = kwargs.pop('savePath', None)

        verbose: bool = kwargs.pop('verbose', False)
        if debug.ON:
            verbose: bool = True

        if loadPath:   # pylint: disable=too-many-nested-blocks
            if verbose:
                self.stdOutLogger.info(msg=(msg := ('Loading & Applying Data Transformations '
                                                    f'from "{loadPath}"...')))
                tic: float = time.time()

            loadPath: Path = Path(loadPath).resolve(strict=True)

            if loadPath in self._PREP_CACHE:
                if debug.ON:
                    self.stdOutLogger.debug(
                        msg=f'*** RETRIEVING PREPROCESSING CACHE FROM {loadPath} ***')

                prepCache: Namespace = self._PREP_CACHE[loadPath]

                catOrigToPrepColMap: Namespace = prepCache.catOrigToPrepColMap
                numOrigToPrepColMap: Namespace = prepCache.numOrigToPrepColMap

            else:
                self._PREP_CACHE[loadPath] = \
                    Namespace(
                        catOrigToPrepColMap=Namespace.from_yaml(
                            loadPath / PandasMLPreprocessor._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME),
                        numOrigToPrepColMap=Namespace.from_yaml(
                            loadPath / PandasMLPreprocessor._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME))

        else:
            cols: Set[str] = {col
                              for col in ((set(cols) & self.possibleFeatureCols)
                                          if cols
                                          else self.possibleFeatureCols)
                              if self.suffNonNull(col)}

            if cols:
                profile: Namespace = self.profile(*cols,
                                                  profileCat=True,
                                                  profileNum=False,
                                                  skipIfInsuffNonNull=True,
                                                  asDict=True,
                                                  verbose=verbose)

            else:
                return self.copy()

            cols: Set[str] = {col
                              for col in cols
                              if ((distinctProportionsIndex :=
                                   profile[col].distinctProportions.index).notnull() &
                                  (distinctProportionsIndex != '')).sum() > 1}

            if not cols:
                return self.copy()

            catCols: Set[str] = {col
                                 for col in ((cols & self.possibleCatCols) - forceNum)
                                 if (col in forceCat) or
                                    (profile[col].distinctProportions
                                     .iloc[:self._maxNCats[col]].sum()
                                     >= self._minProportionByMaxNCats[col])}

            numCols: Set[str] = {col for col in (cols - catCols) if self.typeIsNum(col)}

            cols: Set[str] = catCols | numCols

            if verbose:
                self.stdOutLogger.info(msg=(msg := ('Preprocessing Columns ' +
                                                    ', '.join(f'"{col}"' for col in cols) +
                                                    '...')))
                tic: float = time.time()

            catOrigToPrepColMap: Namespace = Namespace(__SCALE__=scaleCat)

            if catCols:
                if verbose:
                    self.stdOutLogger.info(
                        msg=(cat_prep_msg := ('Transforming Categorical Features ' +
                                              ', '.join(f'"{catCol}"' for catCol in catCols) +
                                              '...')))
                    cat_prep_tic: float = time.time()

                catIdxCols: Set[str] = set()

                if scaleCat:
                    catScaledIdxCols: Set[str] = set()

                for catCol in catCols:
                    catIdxCol: str = self._CAT_IDX_PREFIX + catCol

                    catColType: DataType = self.type(catCol)

                    if is_boolean(catColType):
                        cats: Tuple[bool] = False, True
                        nCats: int = 2

                    else:
                        isStr: bool = is_string(catColType)

                        cats: Tuple[PyPossibleFeatureType] = tuple(
                            cat
                            for cat in
                            (profile[catCol].distinctProportions.index
                             if catCol in forceCat
                             else (profile[catCol].distinctProportions
                                   .index[:self._maxNCats[catCol]]))
                            if notnull(cat) and
                            ((cat != '') if isStr else isfinite(cat)))

                        nCats: int = len(cats)

                    if scaleCat:
                        catPrepCol: str = self._MIN_MAX_SCL_PREFIX + catIdxCol
                        catScaledIdxCols.add(catPrepCol)

                    else:
                        catPrepCol: str = catIdxCol
                        catIdxCols.add(catPrepCol)

                    catOrigToPrepColMap[catCol] = catPrepCol, {'cats': cats, 'n-cats': nCats}

                if verbose:
                    cat_prep_toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'{cat_prep_msg} done!   <{cat_prep_toc - cat_prep_tic:,.1f} s>')

            numOrigToPrepColMap: Namespace = Namespace(__SCALER__=scaler)

            if numCols:
                numScaledCols: Set[str] = set()

                if verbose:
                    self.stdOutLogger.info(
                        msg=(num_prep_msg := ('Transforming Numerical Features ' +
                                              ', '.join(f'"{numCol}"' for numCol in numCols) +
                                              '...')))
                    num_prep_tic: float = time.time()

                outlierTails: Optional[Union[str, Dict[str, Optional[str]]]] = \
                    fill.get('outlierTails', {})
                if not isinstance(outlierTails, DICT_OR_NAMESPACE_TYPES):
                    outlierTails: Dict[str, Optional[str]] = {col: outlierTails for col in numCols}

                _, numNullFillDetails = self.fillNumNull(*numCols,
                                                         nulls=nulls,
                                                         method=fill.get('method', 'mean'),
                                                         value=fill.get('value'),
                                                         outlierTails=outlierTails,
                                                         returnDetails=True,
                                                         verbose=verbose > 1)

                for numCol in numCols:
                    colOutlierTails: Optional[str] = outlierTails.get(numCol, 'both')

                    excludeLowerTail: bool = colOutlierTails in ('lower', 'both')
                    colMin: PyNumType = (self.outlierRstMin(numCol)
                                         if excludeLowerTail
                                         else self.sampleStat(numCol, stat='min'))

                    excludeUpperTail: bool = colOutlierTails in ('upper', 'both')
                    colMax: PyNumType = (self.outlierRstMax(numCol)
                                         if excludeUpperTail
                                         else self.sampleStat(numCol, stat='max'))

                    if colMin < colMax:
                        numColNullFillDetails: Namespace = numNullFillDetails[numCol][1]

                        numColNulls: Tuple[Optional[PyNumType], Optional[PyNumType]] = \
                            numColNullFillDetails['nulls']

                        numColNullFillMethod: Optional[PyNumType] = \
                            numColNullFillDetails['null-fill-method']

                        numColNullFillValue: Optional[PyNumType] = \
                            numColNullFillDetails['null-fill-value']

                        if scaler:
                            if scaler == 'standard':
                                scaledCol: str = self._STD_SCL_PREFIX + numCol

                                series: Series = self.reprSample[numCol]

                                if colOutlierTails == 'both':
                                    series: Series = series.loc[series.between(left=colMin,
                                                                               right=colMax,
                                                                               inclusive='both')]

                                elif colOutlierTails == 'lower':
                                    series: Series = series.loc[series > colMin]

                                elif colOutlierTails == 'upper':
                                    series: Series = series.loc[series < colMax]

                                stdDev: float = float(series.std(axis='index',
                                                                 skipna=True,
                                                                 level=None,
                                                                 ddof=1))

                                numOrigToPrepColMap[numCol] = \
                                    (scaledCol,
                                     {'nulls': numColNulls,
                                      'null-fill-method': numColNullFillMethod,
                                      'null-fill-value': numColNullFillValue,
                                      'mean': numColNullFillValue,
                                      'std': stdDev})

                            elif scaler == 'maxabs':
                                scaledCol: str = self._MAX_ABS_SCL_PREFIX + numCol

                                maxAbs: PyNumType = max(abs(colMin), abs(colMax))

                                numOrigToPrepColMap[numCol] = \
                                    (scaledCol,
                                     {'nulls': numColNulls,
                                      'null-fill-method': numColNullFillMethod,
                                      'null-fill-value': numColNullFillValue,
                                      'max-abs': maxAbs})

                            elif scaler == 'minmax':
                                scaledCol: str = self._MIN_MAX_SCL_PREFIX + numCol

                                numOrigToPrepColMap[numCol] = \
                                    (scaledCol,
                                     {'nulls': numColNulls,
                                      'null-fill-method': numColNullFillMethod,
                                      'null-fill-value': numColNullFillValue,
                                      'orig-min': colMin, 'orig-max': colMax,
                                      'target-min': -1, 'target-max': 1})

                            else:
                                raise ValueError('*** Scaler must be one of '
                                                 '"standard", "maxabs", "minmax" '
                                                 'and None ***')

                        else:
                            scaledCol: str = self._NULL_FILL_PREFIX + numCol

                            numOrigToPrepColMap[numCol] = \
                                (scaledCol,
                                 {'nulls': numColNulls,
                                  'null-fill-method': numColNullFillMethod,
                                  'null-fill-value': numColNullFillValue})

                        numScaledCols.add(scaledCol)

                if verbose:
                    num_prep_toc: float = time.time()
                    self.stdOutLogger.info(
                        msg=f'{num_prep_msg} done!   <{num_prep_toc - num_prep_tic:,.1f} s>')

        if savePath:
            if verbose:
                self.stdOutLogger.info(
                    msg=(prep_save_msg := ('Saving Data Transformations '
                                           f'to Local Path "{savePath}"...')))
                prep_save_tic: float = time.time()

            savePath: Path = Path(savePath).resolve(strict=True)

            catOrigToPrepColMap.to_yaml(
                savePath / PandasMLPreprocessor._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME)
            numOrigToPrepColMap.to_yaml(
                savePath / PandasMLPreprocessor._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME)

            if verbose:
                prep_save_toc: float = time.time()
                self.stdOutLogger.info(
                    msg=f'{prep_save_msg} done!   <{prep_save_toc - prep_save_tic:,.1f} s>')

            self._PREP_CACHE[savePath] = Namespace(catOrigToPrepColMap=catOrigToPrepColMap,
                                                   numOrigToPrepColMap=numOrigToPrepColMap)

        if returnNumPy:
            returnNumPyForCols: Tuple[str] = tuple(
                sorted(catPrepColDetails[0]
                       for catCol, catPrepColDetails in catOrigToPrepColMap.items()
                       if (catCol != '__SCALE__') and
                       isinstance(catPrepColDetails, PY_LIST_OR_TUPLE) and
                       (len(catPrepColDetails) == 2))
                +
                sorted(numPrepColDetails[0]
                       for numCol, numPrepColDetails in numOrigToPrepColMap.items()
                       if (numCol != '__SCALER__') and
                       isinstance(numPrepColDetails, PY_LIST_OR_TUPLE) and
                       (len(numPrepColDetails) == 2)))

        else:
            colsToKeep: Set[str] = self.indexCols | (

                (set(chain.from_iterable(
                     (catCol, catPrepColDetails[0])
                     for catCol, catPrepColDetails in catOrigToPrepColMap.items()
                     if (catCol != '__SCALE__') and
                     isinstance(catPrepColDetails, PY_LIST_OR_TUPLE) and
                     (len(catPrepColDetails) == 2)))
                 |
                 set(chain.from_iterable(
                     (numCol, numPrepColDetails[0])
                     for numCol, numPrepColDetails in numOrigToPrepColMap.items()
                     if (numCol != '__SCALER__') and
                     isinstance(numPrepColDetails, PY_LIST_OR_TUPLE) and
                     (len(numPrepColDetails) == 2))))

                if loadPath

                else (((catCols | (catScaledIdxCols if scaleCat else catIdxCols))
                       if catCols
                       else set())
                      |
                      ((numCols | numScaledCols)
                       if numCols
                       else set())))

        missingCatCols: Set[str] = set(catOrigToPrepColMap) - self.columns - {'__SCALE__'}
        missingNumCols: Set[str] = set(numOrigToPrepColMap) - self.columns - {'__SCALER__'}
        missingCols: Set[str] = missingCatCols | missingNumCols

        addCols: Namespace = Namespace()

        if missingCols:
            if debug.ON:
                self.stdOutLogger.debug(
                    msg=f'*** FILLING MISSING COLS {missingCols} ***')

            for missingCol in missingCols:
                addCols[missingCol] = (NA
                                       if missingCol in missingCatCols
                                       else numOrigToPrepColMap[missingCol][1]['null-fill-value'])

                if not returnNumPy:
                    colsToKeep.add(missingCol)

        s3ParquetDF: S3ParquetDataFeeder = \
            self.map(
                PandasMLPreprocessor(
                    addCols=addCols,
                    typeStrs=Namespace(**{catCol: str(self.type(catCol))
                                          for catCol in (set(catOrigToPrepColMap)
                                                         - {'__SCALE__'})}),
                    catOrigToPrepColMap=catOrigToPrepColMap,
                    numOrigToPrepColMap=numOrigToPrepColMap,
                    returnNumPyForCols=returnNumPyForCols if returnNumPy else None),
                inheritNRows=True, **kwargs)

        if not returnNumPy:
            s3ParquetDF: S3ParquetDataFeeder = s3ParquetDF[tuple(colsToKeep)]
            s3ParquetDF._inheritCache(self, *(() if loadPath else colsToKeep))
            s3ParquetDF._cache.reprSample = self._cache.reprSample

        if verbose:
            toc: float = time.time()
            self.stdOutLogger.info(msg=f'{msg} done!   <{(toc - tic) / 60:,.1f} m>')

        return ((s3ParquetDF, catOrigToPrepColMap, numOrigToPrepColMap)
                if returnOrigToPrepColMaps
                else s3ParquetDF)
