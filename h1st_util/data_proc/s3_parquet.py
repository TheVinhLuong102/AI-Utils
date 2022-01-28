"""S3 Parquet Data Feeder."""


from __future__ import annotations

import datetime
from functools import cache
import json
from logging import Logger
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Optional, Union
from typing import Collection, Dict, List, Set, Sequence, Tuple   # Py3.9+: use built-ins
from urllib.parse import ParseResult, urlparse
from uuid import uuid4

import botocore
import boto3
from numpy import array, allclose, cumsum, hstack, isfinite, isnan, nan, ndarray, vstack
from pandas import DataFrame, Series, concat, isnull, notnull, read_parquet
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from tqdm import tqdm

from pyarrow.dataset import dataset
from pyarrow.fs import S3FileSystem
from pyarrow.lib import RecordBatch, Schema, Table   # pylint: disable=no-name-in-module
from pyarrow.parquet import FileMetaData, read_metadata, read_schema, read_table

from .. import debug, fs, s3
from ..data_types.arrow import (
    DataType, _ARROW_STR_TYPE, _ARROW_DATE_TYPE,
    is_binary, is_boolean, is_complex, is_num, is_possible_cat, is_string)
from ..data_types.numpy_pandas import NUMPY_FLOAT_TYPES, NUMPY_INT_TYPES, PY_NUM_TYPES
from ..data_types.spark_sql import _STR_TYPE
from ..default_dict import DefaultDict
from ..iter import to_iterable
from ..namespace import Namespace

from ._abstract import AbstractDataHandler, ReducedDataSetType


__all__ = ('S3ParquetDataFeeder',)


# flake8: noqa
# (too many camelCase names)

# pylint: disable=c-extension-no-member
# e.g., `math.`

# pylint: disable=invalid-name
# e.g., camelCase names

# pylint: disable=logging-fstring-interpolation,logging-not-lazy

# pylint: disable=no-member
# e.g., `._cache`

# pylint: disable=protected-access
# e.g., `._cache`

# pylint: disable=too-many-lines
# (this whole module)


# pylint: disable=consider-using-f-string
# pylint: disable=too-few-public-methods


class AbstractS3FileDataHandler(AbstractDataHandler):
    # pylint: disable=abstract-method
    """Abstract S3 File Data Handler."""

    S3_CLIENT = boto3.client(service_name='s3',
                             region_name=None,
                             api_version=None,
                             use_ssl=True,
                             verify=None,
                             endpoint_url=None,
                             aws_access_key_id=None,
                             aws_secret_access_key=None,
                             aws_session_token=None,
                             config=botocore.client.Config(connect_timeout=9,
                                                           read_timeout=9))

    _SCHEMA_MIN_N_PIECES: int = 10
    _REPR_SAMPLE_MIN_N_PIECES: int = 100

    @property
    def reprSampleMinNPieces(self) -> int:
        """Minimum number of pieces for reprensetative sample."""
        return self._reprSampleMinNPieces

    @reprSampleMinNPieces.setter
    def reprSampleMinNPieces(self, n: int, /):
        if (n <= self.nPieces) and (n != self._reprSampleMinNPieces):
            self._reprSampleMinNPieces: int = n

    @reprSampleMinNPieces.deleter
    def reprSampleMinNPieces(self):
        self._reprSampleMinNPieces: int = min(self._REPR_SAMPLE_MIN_N_PIECES,
                                              self.nPieces)


class _S3ParquetDataFeeder__getitem__pandasDFTransform:
    def __init__(self, item):
        if isinstance(item, str):
            self.col = item
            self.cols = None

        else:
            self.cols = item
            self.col_set = set(item)

    def __call__(self, pandasDF):
        if self.cols:
            for missingCol in self.col_set.difference(pandasDF.columns):
                pandasDF.loc[:, missingCol] = None

            return pandasDF[self.cols]

        if self.col in pandasDF.columns:
            return pandasDF[self.col]

        return Series(index=pandasDF.index, name=self.col)


class _S3ParquetDataFeeder__drop__pandasDFTransform:
    def __init__(self, cols):
        self.cols = list(cols)

    def __call__(self, pandasDF):
        return pandasDF.drop(columns=self.cols,
                             level=None,
                             inplace=False,
                             errors='ignore')


class _S3ParquetDataFeeder__fillna__pandasDFTransform:
    def __init__(self, nullFillDetails):
        self.nullFillDetails = nullFillDetails

    def __call__(self, pandasDF):
        for col, nullFillColNameNDetails in self.nullFillDetails.items():
            if (col not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and \
                    isinstance(nullFillColNameNDetails, list) and \
                    (len(nullFillColNameNDetails) == 2):
                _, nullFill = nullFillColNameNDetails

                lowerNull, upperNull = nullFill['Nulls']

                series = pandasDF[col]

                chks = series.notnull()

                if lowerNull is not None:
                    chks &= (series > lowerNull)

                if upperNull is not None:
                    chks &= (series < upperNull)

                pandasDF.loc[
                    :,
                    (AbstractDataHandler._NULL_FILL_PREFIX +
                     col +
                     AbstractDataHandler._PREP_SUFFIX)] = \
                    series.where(
                        cond=chks,
                        other=nullFill['NullFillValue'],
                        inplace=False,
                        axis=None,
                        level=None,
                        errors='raise',
                        try_cast=False)
                # ^^^ SettingWithCopyWarning (?)
                # A value is trying to be set
                # on a copy of a slice from a DataFrame.
                # Try using .loc[row_indexer,col_indexer] = value instead

        return pandasDF


class _S3ParquetDataFeeder__prep__pandasDFTransform:
    def __init__(self,
                 addCols,
                 typeStrs,
                 catOrigToPrepColMap,
                 numOrigToPrepColMap,
                 returnNumPyForCols=None):
        self.addCols = addCols

        self.typeStrs = typeStrs

        assert not catOrigToPrepColMap['__OHE__']
        self.catOrigToPrepColMap = catOrigToPrepColMap
        self.scaleCat = catOrigToPrepColMap['__SCALE__']

        self.numNullFillPandasDFTransform = \
            _S3ParquetDataFeeder__fillna__pandasDFTransform(
                nullFillDetails=numOrigToPrepColMap)

        self.numNullFillCols = []
        self.numPrepCols = []
        self.numPrepDetails = []

        for numCol, numPrepColNDetails in numOrigToPrepColMap.items():
            if (numCol not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and \
                    isinstance(numPrepColNDetails, list) and \
                    (len(numPrepColNDetails) == 2):
                self.numNullFillCols.append(
                    AbstractDataHandler._NULL_FILL_PREFIX +
                    numCol +
                    AbstractDataHandler._PREP_SUFFIX)

                numPrepCol, numPrepDetails = numPrepColNDetails
                self.numPrepCols.append(numPrepCol)
                self.numPrepDetails.append(numPrepDetails)

        if returnNumPyForCols:
            self.returnNumPyForCols = \
                to_iterable(returnNumPyForCols, iterable_type=list)

            nCatCols = len(catOrigToPrepColMap)
            self.catPrepCols = returnNumPyForCols[:nCatCols]

            numPrepCols = returnNumPyForCols[nCatCols:]
            numPrepColListIndices = \
                [numPrepCols.index(numPrepCol)
                 for numPrepCol in self.numPrepCols]
            self.numNullFillCols = \
                [self.numNullFillCols[i]
                 for i in numPrepColListIndices]
            self.numPrepCols = \
                [self.numPrepCols[i]
                 for i in numPrepColListIndices]
            self.numPrepDetails = \
                [self.numPrepDetails[i]
                 for i in numPrepColListIndices]

        else:
            self.returnNumPyForCols = None

        self.numScaler = numOrigToPrepColMap['__SCALER__']

        if self.numScaler == 'standard':
            self.numScaler = \
                StandardScaler(
                    copy=True,
                    with_mean=True,
                    with_std=True)

            # mean value for each feature in the training set
            self.numScaler.mean_ = \
                array(
                    [numPrepDetails['Mean']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature relative scaling of the data
            self.numScaler.scale_ = \
                array(
                    [numPrepDetails['StdDev']
                     for numPrepDetails in self.numPrepDetails])

        elif self.numScaler == 'maxabs':
            self.numScaler = \
                MaxAbsScaler(
                    copy=True)

            # per-feature maximum absolute value /
            # per-feature relative scaling of the data
            self.numScaler.max_abs_ = \
                self.numScaler.scale_ = \
                array(
                    [numPrepDetails['MaxAbs']
                     for numPrepDetails in self.numPrepDetails])

        elif self.numScaler == 'minmax':
            self.numScaler = \
                MinMaxScaler(
                    feature_range=(-1, 1),
                    copy=True)

            # per-feature minimum seen in the data
            self.numScaler.data_min_ = \
                array(
                    [numPrepDetails['OrigMin']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature maximum seen in the data
            self.numScaler.data_max_ = \
                array(
                    [numPrepDetails['OrigMax']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature range (data_max_ - data_min_) seen in the data
            self.numScaler.data_range_ = \
                self.numScaler.data_max_ - self.numScaler.data_min_

            # per-feature relative scaling of the data
            self.numScaler.scale_ = \
                2 / self.numScaler.data_range_

            # per-feature adjustment for minimum
            self.numScaler.min_ = \
                -1 - (self.numScaler.scale_ * self.numScaler.data_min_)

        else:
            assert self.numScaler is None

        if self.numScaler is not None:
            self.numScaler.n_features_in_ = len(self.numPrepDetails)

    def __call__(self, pandasDF):
        _FLOAT_ABS_TOL = 1e-9

        for col, value in self.addCols.items():
            pandasDF[col] = value

        for catCol, prepCatColNameNDetails in self.catOrigToPrepColMap.items():
            if (catCol not in ('__OHE__', '__SCALE__')) and \
                    isinstance(prepCatColNameNDetails, list) and \
                    (len(prepCatColNameNDetails) == 2):
                prepCatCol, catColDetails = prepCatColNameNDetails

                cats = catColDetails['Cats']
                nCats = catColDetails['NCats']

                s = pandasDF[catCol]

                pandasDF.loc[:, prepCatCol] = \
                    (sum(((s == cat) * i)
                         for i, cat in enumerate(cats)) +
                     ((~s.isin(cats)) * nCats)) \
                    if self.typeStrs[catCol] == _STR_TYPE \
                    else (sum(((s - cat).abs().between(left=0,
                                                       right=_FLOAT_ABS_TOL,
                                                       inclusive='both') * i)
                              for i, cat in enumerate(cats)) +
                          ((1 -
                            sum((s - cat).abs().between(left=0,
                                                        right=_FLOAT_ABS_TOL,
                                                        inclusive='both')
                                for cat in cats)) *
                           nCats))
                # *** NOTE NumPy BUG ***
                # *** abs(...) of a data type most negative value equals to
                # the same most negative value ***
                # https://github.com/numpy/numpy/issues/5657
                # https://github.com/numpy/numpy/issues/9463
                # http://numpy-discussion.10968.n7.nabble.com/abs-for-max-negative-integers-desired-behavior-td8939.html

                # ^^^ SettingWithCopyWarning (?)
                # A value is trying to be set on
                # a copy of a slice from a DataFrame.
                # Try using .loc[row_indexer,col_indexer] = value instead

                if self.scaleCat:
                    pandasDF.loc[:, prepCatCol] = minMaxScaledIdxSeries = \
                        2 * pandasDF[prepCatCol] / nCats - 1
                    # ^^^ SettingWithCopyWarning (?)
                    # A value is trying to be set on
                    # a copy of a slice from a DataFrame.
                    # Try using .loc[row_indexer,col_indexer] = value instead

                    assert minMaxScaledIdxSeries.between(
                        left=-1, right=1, inclusive='both').all(), \
                        (f'*** "{prepCatCol}" ({nCats:,} CATS) '
                         'CERTAIN MIN-MAX SCALED INT INDICES '
                         'NOT BETWEEN -1 AND 1: '
                         f'({minMaxScaledIdxSeries.min()}, '
                         f'{minMaxScaledIdxSeries.max()}) ***')

        pandasDF = \
            self.numNullFillPandasDFTransform(
                pandasDF=pandasDF)

        if self.returnNumPyForCols:
            return (hstack(
                    (pandasDF[self.catPrepCols].values,
                     self.numScaler.transform(
                         X=pandasDF[self.numNullFillCols])))
                    if self.numScaler
                    else pandasDF[self.returnNumPyForCols].values)

        if self.numScaler:
            pandasDF[self.numPrepCols] = \
                DataFrame(
                    data=self.numScaler.transform(
                        X=pandasDF[self.numNullFillCols]))
            # ^^^ SettingWithCopyWarning (?)
            # A value is trying to be set
            # on a copy of a slice from a DataFrame.
            # Try using .loc[row_indexer,col_indexer] = value instead

        return pandasDF


def randomSample(population: Collection[Any], sampleSize: int,
                 returnCollectionType=set) -> Collection[Any]:
    """Draw random sample from population."""
    return returnCollectionType(random.sample(population=population, k=sampleSize)
                                if len(population) > sampleSize
                                else population)


class S3ParquetDataFeeder(AbstractS3FileDataHandler):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """S3 Parquet Data Feeder."""

    _CACHE: Dict[str, Namespace] = {}

    _PIECE_CACHES: Dict[str, Namespace] = {}

    # default arguments dict
    _DEFAULT_KWARGS = dict(
        iCol=AbstractS3FileDataHandler._DEFAULT_I_COL,
        tCol=None,

        reprSampleMinNPieces=AbstractS3FileDataHandler._REPR_SAMPLE_MIN_N_PIECES,
        reprSampleSize=AbstractS3FileDataHandler._DEFAULT_REPR_SAMPLE_SIZE,

        nulls=DefaultDict((None, None)),
        minNonNullProportion=DefaultDict(
            AbstractS3FileDataHandler._DEFAULT_MIN_NON_NULL_PROPORTION),
        outlierTailProportion=DefaultDict(
            AbstractS3FileDataHandler._DEFAULT_OUTLIER_TAIL_PROPORTION),
        maxNCats=DefaultDict(AbstractS3FileDataHandler._DEFAULT_MAX_N_CATS),
        minProportionByMaxNCats=DefaultDict(
            AbstractS3FileDataHandler._DEFAULT_MIN_PROPORTION_BY_MAX_N_CATS),
    )

    def __init__(self, path: str, *, reCache: bool = False,
                 awsRegion: Optional[str] = None,
                 _mappers: Optional[Union[callable, Sequence[callable]]] = None,
                 verbose: bool = True, **kwargs: Any):
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Init S3 Parquet Data Feeder."""
        if verbose or debug.ON:
            logger: Logger = self.classStdOutLogger()

        assert isinstance(path, str) and path.startswith('s3://'), \
            ValueError(f'*** {path} NOT AN S3 PATH ***')
        self.path: str = path

        self.awsRegion: Optional[str] = awsRegion

        if (not reCache) and (path in self._CACHE):
            _cache = self._CACHE[path]
        else:
            self._CACHE[path] = _cache = Namespace()

        if _cache:
            if debug.ON:
                logger.debug(msg=f'*** RETRIEVING CACHE FOR "{path}" ***')

        else:
            _parsedURL: ParseResult = urlparse(url=path, scheme='', allow_fragments=True)
            _cache.s3Bucket = _parsedURL.netloc
            _cache.pathS3Key = _parsedURL.path[1:]

            _cache.tmpDirS3Key = 'tmp'
            _cache.tmpDirPath = f's3://{_cache.s3Bucket}/{_cache.tmpDirS3Key}'

            if path in self._PIECE_CACHES:
                _cache.nPieces = 1
                _cache.piecePaths = {path}

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

                if _file_paths := _cache._srcArrowDS.files:
                    _cache.piecePaths = {f's3://{file_path}'
                                         for file_path in _file_paths
                                         if not file_path.endswith('_$folder$')}
                    _cache.nPieces = len(_cache.piecePaths)

                else:
                    _cache.nPieces = 1
                    _cache.piecePaths = {path}

            _cache.srcColsInclPartitionKVs = set()
            _cache.srcTypesInclPartitionKVs = Namespace()

            for i, piecePath in enumerate(_cache.piecePaths):
                if piecePath in self._PIECE_CACHES:
                    pieceCache: Namespace = self._PIECE_CACHES[piecePath]

                    if (pieceCache.nRows is None) and (i < self._SCHEMA_MIN_N_PIECES):
                        pieceCache.localPath = self.pieceLocalPath(piecePath=piecePath)

                        schema: Schema = read_schema(where=pieceCache.localPath)

                        pieceCache.srcColsExclPartitionKVs = set(schema.names)

                        pieceCache.srcColsInclPartitionKVs.update(schema.names)

                        for col in (pieceCache.srcColsExclPartitionKVs
                                    .difference(pieceCache.partitionKVs)):
                            pieceCache.srcTypesExclPartitionKVs[col] = \
                                pieceCache.srcTypesInclPartitionKVs[col] = \
                                schema.field(col).type

                        metadata: FileMetaData = read_metadata(where=pieceCache.localPath)
                        pieceCache.nCols = metadata.num_columns
                        pieceCache.nRows = metadata.num_rows

                else:
                    srcColsInclPartitionKVs: Set[str] = set()

                    srcTypesExclPartitionKVs: Namespace = Namespace()
                    srcTypesInclPartitionKVs: Namespace = Namespace()

                    partitionKVs: Dict[str, Union[datetime.date, str]] = {}

                    for partitionKV in re.findall(pattern='[^/]+=[^/]+/', string=piecePath):
                        k, v = partitionKV.split(sep='=', maxsplit=1)

                        srcColsInclPartitionKVs.add(k)

                        if k == self._DEFAULT_D_COL:
                            srcTypesInclPartitionKVs[k] = _ARROW_DATE_TYPE
                            partitionKVs[k] = datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date()

                        else:
                            srcTypesInclPartitionKVs[k] = _ARROW_STR_TYPE
                            partitionKVs[k] = v[:-1]

                    if i < self._SCHEMA_MIN_N_PIECES:
                        localPath: Path = self.pieceLocalPath(piecePath=piecePath)

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

                        srcColsExclPartitionKVs: Optional[List[str]] = None

                        nCols: Optional[int] = None
                        nRows: Optional[int] = None

                    self._PIECE_CACHES[piecePath] = pieceCache = \
                        Namespace(
                            localPath=localPath,
                            partitionKVs=partitionKVs,

                            srcColsExclPartitionKVs=srcColsExclPartitionKVs,
                            srcColsInclPartitionKVs=srcColsInclPartitionKVs,

                            srcTypesExclPartitionKVs=srcTypesExclPartitionKVs,
                            srcTypesInclPartitionKVs=srcTypesInclPartitionKVs,

                            nCols=nCols, nRows=nRows)

                _cache.srcColsInclPartitionKVs |= pieceCache.srcColsInclPartitionKVs

                for col, arrowType in pieceCache.srcTypesInclPartitionKVs.items():
                    if col in _cache.srcTypesInclPartitionKVs:
                        assert arrowType == _cache.srcTypesInclPartitionKVs[col], \
                            TypeError(f'*** {piecePath} COLUMN {col}: '
                                      f'DETECTED TYPE {arrowType} != '
                                      f'{_cache.srcTypesInclPartitionKVs[col]} ***')
                    else:
                        _cache.srcTypesInclPartitionKVs[col] = arrowType

        self.__dict__.update(_cache)

        self._cachedLocally: bool = False

        self._mappers: Tuple[callable] = (()
                                          if _mappers is None
                                          else to_iterable(_mappers,
                                                           iterable_type=tuple))

        # extract standard keyword arguments
        self._extractStdKwArgs(kwargs, resetToClassDefaults=True, inplace=True)

        # organize time series if applicable
        self._organizeTimeSeries()

        # set profiling settings and create empty profiling cache
        self._emptyCache()

    # ================================
    # "INTERNAL / DON'T TOUCH" METHODS
    # --------------------------------
    # _extractStdKwArgs
    # _organizeTimeSeries
    # _emptyCache
    # _inheritCache

    # pylint: disable=inconsistent-return-statements
    def _extractStdKwArgs(self, kwargs: Dict[str, Any], /, *,
                          resetToClassDefaults: bool = False,
                          inplace: bool = False) -> Optional[Namespace]:
        namespace: Namespace = self if inplace else Namespace()

        for k, classDefaultV in self._DEFAULT_KWARGS.items():
            _privateK: str = f'_{k}'

            if not resetToClassDefaults:
                existingInstanceV = getattr(self, _privateK, None)

            v = kwargs.pop(k,
                           existingInstanceV
                           if (not resetToClassDefaults) and existingInstanceV
                           else classDefaultV)

            if (k == 'reprSampleMinNPieces') and (v > self.nPieces):
                v = self.nPieces

            setattr(namespace,
                    _privateK   # USE _k TO NOT INVOKE @k.setter RIGHT AWAY
                    if inplace
                    else k,
                    v)

        if inplace:
            if self._iCol not in self.columns:
                self._iCol = None

            if self._tCol not in self.columns:
                self._tCol = None

        else:
            return namespace

    def _organizeTimeSeries(self):
        self._dCol: Optional[str] = (self._DEFAULT_D_COL
                                     if self._DEFAULT_D_COL in self.columns
                                     else None)

        self.hasTS: bool = bool(self._iCol and self._tCol)

    def _emptyCache(self):
        self._cache: Namespace = \
            Namespace(prelimReprSamplePiecePaths=None,
                      reprSamplePiecePaths=None,
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

            for sameCol in (commonCols
                            .difference(newColToOldColMap)
                            .intersection(sameCols)):
                newColToOldColMap[sameCol] = sameCol

        else:
            newColToOldColMap: Dict[str, str] = {col: col for col in commonCols}

        for cacheCategory in ('count', 'distinct',
                              'nonNullProportion',
                              'suffNonNullProportionThreshold',
                              'suffNonNull',
                              'sampleMin', 'sampleMax',
                              'sampleMean', 'sampleMedian',
                              'outlierRstMin', 'outlierRstMax',
                              'outlierRstMean', 'outlierRstMedian'):
            for newCol, oldCol in newColToOldColMap.items():
                if oldCol in oldS3ParquetDF._cache.__dict__[cacheCategory]:
                    self._cache.__dict__[cacheCategory][newCol] = \
                        oldS3ParquetDF._cache.__dict__[cacheCategory][oldCol]

    # ========
    # ITERATOR
    # --------
    # __iter__
    # __next__

    def __iter__(self) -> S3ParquetDataFeeder:
        """Iterate through pieces."""
        s3_parquet_df = self.copy(inheritCache=True, inheritNRows=True)

        # pylint: disable=attribute-defined-outside-init
        s3_parquet_df.piecePathsToIter = s3_parquet_df.piecePaths.copy()

        return s3_parquet_df

    def __next__(self) -> ReducedDataSetType:
        """Iterate through next piece."""
        if self.piecePathsToIter:
            return self.reduce(self.piecePathsToIter.pop(), verbose=False)

        raise StopIteration

    # ====
    # COPY
    # ----
    # copy

    def copy(self, **kwargs: Any) -> S3ParquetDataFeeder:
        """Make a copy."""
        resetMappers: bool = kwargs.pop('resetMappers', False)
        inheritCache: bool = kwargs.pop('inheritCache', not resetMappers)
        inheritNRows: bool = kwargs.pop('inheritNRows', inheritCache)

        s3ParquetDF: S3ParquetDataFeeder = \
            S3ParquetDataFeeder(
                path=self.path, awsRegion=self.awsRegion,

                iCol=self._iCol, tCol=self._tCol,

                _mappers=() if resetMappers else self._mappers,

                reprSampleMinNPieces=self._reprSampleMinNPieces,
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

    # ===============
    # STRING REPR/STR
    # ---------------
    # __repr__
    # __short_repr__

    def __repr__(self) -> str:
        """Return string repr."""
        colAndTypeStrs: List[str] = []

        if self._iCol:
            colAndTypeStrs.append(f'(iCol) {self._iCol}: {self.type(self._iCol)}')

        if self._dCol:
            colAndTypeStrs.append(f'(dCol) {self._dCol}: {self.type(self._dCol)}')

        if self._tCol:
            colAndTypeStrs.append(f'(tCol) {self._tCol}: {self.type(self._tCol)}')

        colAndTypeStrs.extend(f'{col}: {self.type(col)}'
                              for col in self.contentCols)

        return (f'{self.nPieces:,}-piece ' +
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

        return (f'{self.nPieces:,}-piece ' +
                (f'{self._cache.nRows:,}-row '
                 if self._cache.nRows
                 else (f'approx-{self._cache.approxNRows:,.0f}-row '
                       if self._cache.approxNRows
                       else '')) +
                type(self).__name__ +
                (f'[{self.path} + {len(self._mappers):,} transform(s)]'
                 f"[{', '.join(colsDescStr)}]"))

    # ===============
    # CACHING METHODS
    # ---------------
    # cacheLocally
    # pieceLocalPath

    def cacheLocally(self, verbose: bool = True):
        """Cache files to local disk."""
        if not self._cachedLocally:
            if verbose:
                self.stdOutLogger.info(msg=(msg := 'Caching Files to Local Disk...'))

            parsedURL: ParseResult = urlparse(url=self.path, scheme='', allow_fragments=True)

            localPath: str = str(self._TMP_DIR_PATH / parsedURL.netloc / parsedURL.path[1:])

            s3.sync(from_dir_path=self.path,
                    to_dir_path=localPath,
                    delete=True, quiet=True,
                    verbose=True)

            for piecePath in self.piecePaths:
                self._PIECE_CACHES[piecePath].localPath = \
                    piecePath.replace(self.path, localPath)

            self._cachedLocally: bool = True

            if verbose:
                self.stdOutLogger.info(msg=f'{msg} done!')

    def pieceLocalPath(self, piecePath: str) -> Path:
        """Get local cache file path of piece."""
        if (piecePath in self._PIECE_CACHES) and self._PIECE_CACHES[piecePath].localPath:
            return self._PIECE_CACHES[piecePath].localPath

        parsedURL: ParseResult = urlparse(url=piecePath, scheme='', allow_fragments=True)

        localPath: Path = self._TMP_DIR_PATH / parsedURL.netloc / parsedURL.path[1:]

        localDirPath: Path = localPath.parent
        fs.mkdir(dir_path=localDirPath, hdfs=False)
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

        if piecePath in self._PIECE_CACHES:
            self._PIECE_CACHES[piecePath].localPath = localPath

        return localPath

    # ***********************
    # MAP-REDUCE (PARTITIONS)
    # -----------------------
    # map
    # reduce
    # __getitem__
    # drop
    # rename
    # filter
    # collect
    # toPandas

    def map(self,
            mappers: Optional[Union[callable, Sequence[callable]]] = None, /,
            **kwargs: Any) -> S3ParquetDataFeeder:
        """Apply mapper function(s) to pieces."""
        if mappers is None:
            mappers: Tuple[callable] = ()

        inheritCache: bool = kwargs.pop('inheritCache', False)
        inheritNRows: bool = kwargs.pop('inheritNRows', inheritCache)

        s3ParquetDF: S3ParquetDataFeeder = \
            S3ParquetDataFeeder(
                path=self.path,
                awsRegion=self.awsRegion,

                iCol=self._iCol, tCol=self._tCol,
                _mappers=self._mappers + to_iterable(mappers, iterable_type=tuple),

                reprSampleMinNPieces=self._reprSampleMinNPieces,
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

    def reduce(self, *piecePaths: str, **kwargs: Any) -> ReducedDataSetType:
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Reduce from mapped content."""
        _CHUNK_SIZE: int = 10 ** 5

        nSamplesPerPiece: int = kwargs.get('nSamplesPerPiece')

        reducer: callable = kwargs.get(
            'reducer',
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

        if not piecePaths:
            piecePaths: Set[str] = self.piecePaths

        results: List[ReducedDataSetType] = []

        # pylint: disable=too-many-nested-blocks
        for piecePath in (tqdm(piecePaths) if verbose and (len(piecePaths) > 1) else piecePaths):
            pieceLocalPath: Path = self.pieceLocalPath(piecePath=piecePath)

            pieceCache: Namespace = self._PIECE_CACHES[piecePath]

            if pieceCache.nRows is None:
                schema: Schema = read_schema(where=pieceLocalPath)

                pieceCache.srcColsExclPartitionKVs = set(schema.names)

                pieceCache.srcColsInclPartitionKVs.update(schema.names)

                self.srcColsInclPartitionKVs.update(schema.names)

                for col in (pieceCache.srcColsExclPartitionKVs
                            .difference(pieceCache.partitionKVs)):
                    pieceCache.srcTypesExclPartitionKVs[col] = \
                        pieceCache.srcTypesInclPartitionKVs[col] = \
                        _arrowType = schema.field(col).type

                    assert not is_binary(_arrowType), \
                        TypeError(f'*** {piecePath}: {col} IS OF BINARY TYPE ***')

                    if col in self.srcTypesInclPartitionKVs:
                        assert _arrowType == self.srcTypesInclPartitionKVs[col], \
                            TypeError(f'*** {piecePath} COLUMN {col}: '
                                      f'DETECTED TYPE {_arrowType} != '
                                      f'{self.srcTypesInclPartitionKVs[col]} ***')
                    else:
                        self.srcTypesInclPartitionKVs[col] = _arrowType

                metadata: FileMetaData = read_metadata(where=pieceCache.localPath)
                pieceCache.nCols = metadata.num_columns
                pieceCache.nRows = metadata.num_rows

            cols: Optional[Collection[str]] = kwargs.get('cols')

            cols: Set[str] = (to_iterable(cols, iterable_type=set)
                              if cols
                              else pieceCache.srcColsInclPartitionKVs)

            srcCols: Set[str] = cols & pieceCache.srcColsExclPartitionKVs

            partitionKeyCols: Set[str] = cols.intersection(pieceCache.partitionKVs)

            if srcCols:
                pandasDFConstructed: bool = False

                if toSubSample := nSamplesPerPiece and (nSamplesPerPiece < pieceCache.nRows):
                    intermediateN: float = (nSamplesPerPiece * pieceCache.nRows) ** .5

                    if ((nChunksForIntermediateN := int(math.ceil(intermediateN / _CHUNK_SIZE)))
                            < (approxNChunks := int(math.ceil(pieceCache.nRows / _CHUNK_SIZE)))):
                        # arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table
                        pieceArrowTable: Table = read_table(source=pieceLocalPath,
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
                            pieceArrowTable.to_batches(max_chunksize=_CHUNK_SIZE)

                        nChunks: int = len(chunkRecordBatches)

                        assert nChunks in (approxNChunks - 1, approxNChunks), \
                            ValueError(f'*** {piecePath}: {nChunks} vs. '
                                       f'{approxNChunks} Record Batches ***')

                        assert nChunksForIntermediateN <= nChunks, \
                            ValueError(f'*** {piecePath}: {nChunksForIntermediateN} vs. '
                                       f'{nChunks} Record Batches ***')

                        chunkPandasDFs: List[DataFrame] = []

                        nSamplesPerChunk: int = int(math.ceil(nSamplesPerPiece /
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
                                chunkPandasDF[k] = pieceCache.partitionKVs[k]

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

                        piecePandasDF: DataFrame = \
                            concat(objs=chunkPandasDFs,
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
                    piecePandasDF: DataFrame = read_parquet(
                        path=pieceLocalPath,
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
                        piecePandasDF[k] = pieceCache.partitionKVs[k]

                    if toSubSample:
                        piecePandasDF: DataFrame = \
                            piecePandasDF.sample(n=nSamplesPerPiece,
                                                 # frac=None,
                                                 replace=False,
                                                 weights=None,
                                                 random_state=None,
                                                 axis='index',
                                                 ignore_index=False)

            else:
                piecePandasDF: DataFrame = DataFrame(
                    index=range(nSamplesPerPiece
                                if nSamplesPerPiece and
                                (nSamplesPerPiece < pieceCache.nRows)
                                else pieceCache.nRows))

                for k in partitionKeyCols:
                    piecePandasDF[k] = pieceCache.partitionKVs[k]

            result: ReducedDataSetType = piecePandasDF
            for mapper in self._mappers:
                result: ReducedDataSetType = mapper(result)

            results.append(result)

        return reducer(results)

    def __getitem__(self, item: str) -> S3ParquetDataFeeder:
        """Get column."""
        return self.map(
            _S3ParquetDataFeeder__getitem__pandasDFTransform(item=item),
            inheritNRows=True)

    def drop(self, *cols: str, **kwargs: Any) -> S3ParquetDataFeeder:
        """Drop column(s)."""
        return self.map(
            _S3ParquetDataFeeder__drop__pandasDFTransform(cols=cols),
            inheritNRows=True,
            **kwargs)

    def rename(self, **kwargs: Union[str, Any]) -> S3ParquetDataFeeder:
        """Rename data columns (``newColName`` = ``existingColName``)."""
        renameDict: Dict[str, str] = {}
        remainingKwargs: Dict[str, Any] = {}

        for k, v in kwargs.items():
            if v in self.columns:
                renameDict[v] = k
            else:
                remainingKwargs[k] = v

        return self.map(lambda df: df.rename(mapper=None,
                                             index=None,
                                             columns=renameDict,
                                             axis='columns',
                                             copy=False,
                                             inplace=False,
                                             level=None,
                                             errors='ignore'),
                        inheritNRows=True,
                        **remainingKwargs)

    def filter(self, *conditions: str, **kwargs: Any) -> S3ParquetDataFeeder:
        """Apply filtering mapper."""
        s3ParquetDF: S3ParquetDataFeeder = self

        for condition in conditions:
            # pylint: disable=cell-var-from-loop
            s3ParquetDF: S3ParquetDataFeeder = \
                s3ParquetDF.map(lambda df: df.query(expr=condition, inplace=False),
                                **kwargs)

        return s3ParquetDF

    def collect(self, *cols: str, **kwargs: Any) -> ReducedDataSetType:
        """Collect content."""
        return self.reduce(cols=cols if cols else None, **kwargs)

    def toPandas(self, *cols: str, **kwargs: Any) \
            -> Union[DataFrame, Series]:
        """Collect content to Pandas form."""
        return self.collect(*cols, **kwargs)

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

            if iCol is None:
                self.hasTS: bool = False

            else:
                assert iCol, ValueError(f'*** iCol {iCol} INVALID ***')

                self.hasTS: bool = bool(self._tCol)

    @iCol.deleter
    def iCol(self):
        self._iCol: Optional[str] = None

        self.hasTS: bool = False

    @property
    def tCol(self) -> Optional[str]:
        """Date-Time column."""
        return self._tCol

    @tCol.setter
    def tCol(self, tCol: str):
        if tCol != self._tCol:
            self._tCol: Optional[str] = tCol

            if tCol is None:
                self.hasTS: bool = False

            else:
                assert tCol, ValueError(f'*** tCol {tCol} INVALID ***')

                self.hasTS: bool = bool(self._iCol)

    @tCol.deleter
    def tCol(self):
        self._tCol: Optional[str] = None

        self.hasTS: bool = False

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
        self._reprSampleSize = len(self._cache.reprSample)

        self._cache.nonNullProportion = {}
        self._cache.suffNonNull = {}

    # =====================
    # ROWS, COLUMNS & TYPES
    # ---------------------
    # approxNRows
    # nRows
    # __len__
    # columns
    # types
    # type / typeIsNum / typeIsComplex

    def _readMetadataAndSchema(self, piecePath: str) -> Namespace:
        pieceLocalPath: Path = self.pieceLocalPath(piecePath=piecePath)

        pieceCache: Namespace = self._PIECE_CACHES[piecePath]

        if pieceCache.nRows is None:
            schema: Schema = read_schema(where=pieceLocalPath)

            pieceCache.srcColsExclPartitionKVs = set(schema.names)

            pieceCache.srcColsInclPartitionKVs.update(schema.names)

            self.srcColsInclPartitionKVs.update(schema.names)

            for col in (pieceCache.srcColsExclPartitionKVs
                        .difference(pieceCache.partitionKVs)):
                pieceCache.srcTypesExclPartitionKVs[col] = \
                    pieceCache.srcTypesInclPartitionKVs[col] = \
                    _arrowType = schema.field(col).type

                assert not is_binary(_arrowType), \
                    f'*** {piecePath}: {col} IS OF BINARY TYPE ***'

                if col in self.srcTypesInclPartitionKVs:
                    assert _arrowType == self.srcTypesInclPartitionKVs[col], \
                        TypeError(f'*** {piecePath} COLUMN {col}: '
                                  f'DETECTED TYPE {_arrowType} != '
                                  f'{self.srcTypesInclPartitionKVs[col]} ***')
                else:
                    self.srcTypesInclPartitionKVs[col] = _arrowType

            metadata: FileMetaData = read_metadata(where=pieceCache.localPath)
            pieceCache.nCols = metadata.num_columns
            pieceCache.nRows = metadata.num_rows

        return pieceCache

    @property
    def approxNRows(self) -> int:
        """Approximate number of rows."""
        if self._cache.approxNRows is None:
            self.stdOutLogger.info(msg='Counting Approx. No. of Rows...')

            self._cache.approxNRows = (
                self.nPieces
                * sum(self._readMetadataAndSchema(piecePath=piecePath).nRows
                      for piecePath in
                      (tqdm(self.prelimReprSamplePiecePaths)
                       if len(self.prelimReprSamplePiecePaths) > 1
                       else self.prelimReprSamplePiecePaths))
                / self._reprSampleMinNPieces)

        return self._cache.approxNRows

    @property
    def nRows(self) -> int:
        """Return number of rows."""
        if self._cache.nRows is None:
            self.stdOutLogger.info(msg='Counting No. of Rows...')

            self._cache.nRows = \
                sum(self._readMetadataAndSchema(piecePath=piecePath).nRows
                    for piecePath in (tqdm(self.piecePaths)
                                      if self.nPieces > 1
                                      else self.piecePaths))

        return self._cache.nRows

    def __len__(self) -> int:
        """Return (approximate) number of rows."""
        return self._cache.nRows if self._cache.nRows else self.approxNRows

    @property
    def columns(self) -> Set[str]:
        """Column names."""
        return self.srcColsInclPartitionKVs

    @property
    def types(self) -> Namespace:
        """Return column data types."""
        return self.srcTypesInclPartitionKVs

    def type(self, col: str) -> DataType:
        """Return data type of specified column."""
        return self.types[col]

    def typeIsNum(self, col: str) -> bool:
        """Check whether specified column's data type is numerical."""
        return is_num(self.type(col))

    def typeIsComplex(self, col: str) -> bool:
        """Check whether specified column's data type is complex."""
        return is_complex(self.type(col))

    # =============
    # COLUMN GROUPS
    # -------------
    # indexCols
    # possibleFeatureContentCols
    # possibleCatContentCols

    @property
    def indexCols(self) -> Tuple[str]:
        """Return index columns."""
        return (((self._iCol,) if self._iCol else ()) +
                ((self._dCol,) if self._dCol else ()) +
                ((self._tCol,) if self._tCol else ()))

    @property
    def possibleFeatureContentCols(self) -> Set[str]:
        """Possible feature columns for ML modeling."""
        def is_possible_feature(t: DataType) -> bool:
            return is_boolean(t) or is_string(t) or is_num(t)

        return {col for col in self.contentCols if is_possible_feature(self.type(col))}

    @property
    def possibleCatContentCols(self) -> Set[str]:
        """Possible categorical content columns."""
        return {col for col in self.contentCols if is_possible_cat(self.type(col))}

    # ==============
    # SUBSET METHODS
    # --------------
    # _subset
    # filterByPartitionKeys
    # sample
    # gen

    def _subset(self, *piecePaths: str, **kwargs: Any) -> S3ParquetDataFeeder:
        if piecePaths:
            assert self.piecePaths.issuperset(piecePaths)

            nPiecePaths: int = len(piecePaths)

            if nPiecePaths == self.nPieces:
                return self

            if nPiecePaths > 1:
                verbose: bool = kwargs.pop('verbose', True)

                subsetDirS3Key: str = f'{self.tmpDirS3Key}/{uuid4()}'

                _pathPlusSepLen: int = len(self.path) + 1

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

                subsetPath: str = f's3://{self.s3Bucket}/{subsetDirS3Key}'

            else:
                subsetPath: str = piecePaths[0]

            return S3ParquetDataFeeder(
                path=subsetPath, awsRegion=self.awsRegion,

                iCol=self._iCol, tCol=self._tCol,
                _mappers=self._mappers,

                reprSampleMinNPieces=self._reprSampleMinNPieces,
                reprSampleSize=self._reprSampleSize,

                minNonNullProportion=self._minNonNullProportion,
                outlierTailProportion=self._outlierTailProportion,
                maxNCats=self._maxNCats,
                minProportionByMaxNCats=self._minProportionByMaxNCats,

                **kwargs)

        return self

    @cache
    def filterByPartitionKeys(self,
                              *filterCriteriaTuples: Union[Tuple[str, str],
                                                           Tuple[str, str, str]],
                              **kwargs: Any) -> S3ParquetDataFeeder:
        # pylint: disable=too-many-branches
        """Filter by partition keys."""
        filterCriteria: Dict[str, Tuple[Optional[str], Optional[str], Optional[Set[str]]]] = {}

        _samplePiecePath: str = next(iter(self.piecePaths))

        for filterCriteriaTuple in filterCriteriaTuples:
            assert isinstance(filterCriteriaTuple, (list, tuple))
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

                    if ((fromVal is not None) and (v < fromVal)) or \
                            ((toVal is not None) and (v > toVal)) or \
                            ((inSet is not None) and (v not in inSet)):
                        pieceSatisfiesCriteria: bool = False
                        break

                if pieceSatisfiesCriteria:
                    piecePaths.add(piecePath)

            assert piecePaths, \
                FileNotFoundError(f'*** {self}: NO PIECE PATHS SATISFYING '
                                  f'FILTER CRITERIA {filterCriteria} ***')

            if debug.ON:
                self.stdOutLogger.debug(
                    msg=(f'*** {len(piecePaths)} PIECES SATISFYING '
                         f'FILTERING CRITERIA: {filterCriteria} ***'))

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

    def count(self, *cols: str, **kwargs: Any):
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
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**{col: self.count(col, **kwargs)
                                for col in cols})

        col = cols[0]

        pandasDF = kwargs.get('pandasDF')

        lowerNumericNull, upperNumericNull = self._nulls[col]

        if pandasDF is None:
            if col not in self._cache.count:
                verbose = (True
                           if debug.ON
                           else kwargs.get('verbose'))

                if verbose:
                    tic = time.time()

                self._cache.count[col] = result = \
                    self[col] \
                    .map(
                        ((lambda series:
                            series.notnull().sum(skipna=True, min_count=0))
                            if isnull(upperNumericNull)
                            else (lambda series:
                                  (series < upperNumericNull)
                                  .sum(skipna=True, min_count=0)))
                        if isnull(lowerNumericNull)
                        else ((lambda series:
                               (series > lowerNumericNull)
                               .sum(skipna=True, min_count=0))
                              if isnull(upperNumericNull)
                              else (lambda series:
                                    series.between(left=lowerNumericNull,
                                                   right=upperNumericNull,
                                                   inclusive='neither')
                                    .sum(skipna=True, min_count=0)))) \
                    .reduce(
                        cols=col,
                        reducer=sum)

                assert isinstance(result, int), \
                    f'*** "{col}" COUNT = {result} ***'

                if verbose:
                    toc = time.time()
                    self.stdOutLogger.info(
                        msg=(f'No. of Non-NULLs of Column "{col}" = '
                             f'{result:,}   <{toc - tic:,.1f} s>'))

            return self._cache.count[col]

        return (
            (pandasDF[col].notnull().sum(skipna=True, min_count=0)
             if isnull(upperNumericNull)
             else (pandasDF[col] < upperNumericNull).sum(skipna=True,
                                                         min_count=0))

            if isnull(lowerNumericNull)

            else ((pandasDF[col] > lowerNumericNull).sum(skipna=True,
                                                         min_count=0)

                  if isnull(upperNumericNull)

                  else pandasDF[col].between(left=lowerNumericNull,
                                             right=upperNumericNull,
                                             inclusive='neither').sum(
                                                 skipna=True,
                                                 min_count=0))
        )

    def nonNullProportion(self, *cols: str, **kwargs: Any):
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
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**{col: self.nonNullProportion(col, **kwargs)
                                for col in cols})

        col = cols[0]

        if col not in self._cache.nonNullProportion:
            self._cache.nonNullProportion[col] = \
                self.count(
                    col,
                    pandasDF=self.reprSample,
                    **kwargs) \
                / self.reprSampleSize

        return self._cache.nonNullProportion[col]

    def distinct(self, *cols, **kwargs):
        """Return distinct values in specified column(s).

        Return:
            *Approximate* list of distinct values of ``ADF``'s column ``col``,
                with optional descending-sorted counts for those values

        Args:
            col (str): name of a column

            count (bool): whether to count the number of appearances
            of each distinct value of the specified ``col``

            **kwargs:
        """
        if not cols:
            cols = self.contentCols

        asDict = kwargs.pop('asDict', False)

        if len(cols) > 1:
            return Namespace(**{col: self.distinct(col, **kwargs)
                                for col in cols})

        col = cols[0]

        if col not in self._cache.distinct:
            self._cache.distinct[col] = \
                self.reprSample[col].value_counts(
                    normalize=True,
                    sort=True,
                    ascending=False,
                    bins=None,
                    dropna=False)

        return (Namespace(**{col: self._cache.distinct[col]})
                if asDict
                else self._cache.distinct[col])

    @cache
    def quantile(self, *cols: str, **kwargs: Any):
        """Return quantile values in specified column(s)."""
        if len(cols) > 1:
            return Namespace(**{col: self.quantile(col, **kwargs)
                                for col in cols})

        col = cols[0]

        return self[col] \
            .reduce(cols=col) \
            .quantile(
                q=kwargs.get('q', .5),
                interpolation='linear')

    def sampleStat(self, *cols: str, **kwargs: Any) \
            -> Union[Collection, Namespace]:
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
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**{col: self.sampleStat(col, **kwargs)
                                for col in cols})

        col = cols[0]

        if self.typeIsNum(col):
            stat = kwargs.pop('stat', 'mean').lower()
            if stat == 'avg':
                stat = 'mean'
            capitalizedStatName = stat.capitalize()
            s = f'sample{capitalizedStatName}'

            if hasattr(self, s):
                return getattr(self, s)(col, **kwargs)

            if s not in self._cache:
                setattr(self._cache, s, {})
            _cache = getattr(self._cache, s)

            if col not in _cache:
                verbose = True \
                    if debug.ON \
                    else kwargs.get('verbose')

                if verbose:
                    tic = time.time()

                result = \
                    getattr(self.reprSample[col], stat)(
                        axis='index',
                        skipna=True,
                        level=None)

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result = int(result)

                assert isinstance(result, PY_NUM_TYPES), \
                    (f'*** "{col}" SAMPLE '
                        f'{capitalizedStatName.upper()} = '
                        f'{result} ({type(result)}) ***')

                if verbose:
                    toc = time.time()
                    self.stdOutLogger.info(
                        msg=(f'Sample {capitalizedStatName} for '
                             f'Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                _cache[col] = result

            return _cache[col]

        raise ValueError(
            f'{self}.sampleStat({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def outlierRstStat(self, *cols: str, **kwargs: Any):
        # pylint: disable=too-many-branches
        """Return outlier-resistant stat for specified column(s)."""
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**{col: self.outlierRstStat(col, **kwargs)
                                for col in cols})

        col = cols[0]

        if self.typeIsNum(col):
            stat = kwargs.pop('stat', 'mean').lower()
            if stat == 'avg':
                stat = 'mean'
            capitalizedStatName = stat.capitalize()
            s = f'outlierRst{capitalizedStatName}'

            if hasattr(self, s):
                return getattr(self, s)(col, **kwargs)

            if s not in self._cache:
                setattr(self._cache, s, {})
            _cache = getattr(self._cache, s)

            if col not in _cache:
                verbose = True \
                    if debug.ON \
                    else kwargs.get('verbose')

                if verbose:
                    tic = time.time()

                series = self.reprSample[col]

                outlierTails = kwargs.pop('outlierTails', 'both')

                if outlierTails == 'both':
                    series = series.loc[
                        series.between(
                            left=self.outlierRstMin(col),
                            right=self.outlierRstMax(col),
                            inclusive='both')]

                elif outlierTails == 'lower':
                    series = series.loc[
                        series >= self.outlierRstMin(col)]

                elif outlierTails == 'upper':
                    series = series.loc[
                        series <= self.outlierRstMax(col)]

                result = \
                    getattr(series, stat)(
                        axis='index',
                        skipna=True,
                        level=None)

                if isnull(result):
                    self.stdOutLogger.warning(
                        msg=(f'*** "{col}" OUTLIER-RESISTANT '
                             f'{capitalizedStatName.upper()} = '
                             f'{result} ***'))

                    result = self.outlierRstMin(col)

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result = int(result)

                assert isinstance(result, PY_NUM_TYPES), \
                    (f'*** "{col}" '
                        f'OUTLIER-RESISTANT {capitalizedStatName.upper()}'
                        f' = {result} ({type(result)}) ***')

                if verbose:
                    toc = time.time()
                    self.stdOutLogger.info(
                        msg=(f'Outlier-Resistant {capitalizedStatName}'
                             f' for Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                _cache[col] = result

            return _cache[col]

        raise ValueError(
            f'{self}.outlierRstStat({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def outlierRstMin(self, *cols: str, **kwargs: Any):
        """Return outlier-resistant minimum for specified column(s)."""
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**{col: self.outlierRstMin(col, **kwargs)
                                for col in cols})

        col = cols[0]

        if self.typeIsNum(col):
            if 'outlierRstMin' not in self._cache:
                self._cache.outlierRstMin = {}

            if col not in self._cache.outlierRstMin:
                verbose = True \
                    if debug.ON \
                    else kwargs.get('verbose')

                if verbose:
                    tic = time.time()

                series = self.reprSample[col]

                outlierRstMin = \
                    series.quantile(
                        q=self._outlierTailProportion[col],
                        interpolation='linear')

                sampleMin = self.sampleStat(col, stat='min')
                sampleMedian = self.sampleStat(col, stat='median')

                result = (
                    series
                    .loc[series > sampleMin]
                    .min(axis='index', skipna=True, level=None)) \
                    if (outlierRstMin == sampleMin) and \
                    (outlierRstMin < sampleMedian) \
                    else outlierRstMin

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result = int(result)

                assert isinstance(result, PY_NUM_TYPES), \
                    (f'*** "{col}" OUTLIER-RESISTANT MIN = '
                        f'{result} ({type(result)}) ***')

                if verbose:
                    toc = time.time()
                    self.stdOutLogger.info(
                        msg=(f'Outlier-Resistant Min of Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                self._cache.outlierRstMin[col] = result

            return self._cache.outlierRstMin[col]

        raise ValueError(
            f'{self}.outlierRstMin({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def outlierRstMax(self, *cols: str, **kwargs: Any):
        """Return outlier-resistant maximum for specified column(s)."""
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**{col: self.outlierRstMax(col, **kwargs)
                                for col in cols})

        col = cols[0]

        if self.typeIsNum(col):
            if 'outlierRstMax' not in self._cache:
                self._cache.outlierRstMax = {}

            if col not in self._cache.outlierRstMax:
                verbose = (True
                           if debug.ON
                           else kwargs.get('verbose'))

                if verbose:
                    tic = time.time()

                series = self.reprSample[col]

                outlierRstMax = \
                    series.quantile(
                        q=1 - self._outlierTailProportion[col],
                        interpolation='linear')

                sampleMax = self.sampleStat(col, stat='max')
                sampleMedian = self.sampleStat(col, stat='median')

                result = (
                    series
                    .loc[series < sampleMax]
                    .max(axis='index', skipna=True, level=None)) \
                    if (outlierRstMax == sampleMax) and \
                    (outlierRstMax > sampleMedian) \
                    else outlierRstMax

                if isinstance(result, NUMPY_FLOAT_TYPES):
                    result = float(result)

                elif isinstance(result, NUMPY_INT_TYPES):
                    result = int(result)

                assert isinstance(result, PY_NUM_TYPES), \
                    (f'*** "{col}" OUTLIER-RESISTANT MAX = {result} '
                     f'({type(result)}) ***')

                if verbose:
                    toc = time.time()
                    self.stdOutLogger.info(
                        msg=(f'Outlier-Resistant Max of Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                self._cache.outlierRstMax[col] = result

            return self._cache.outlierRstMax[col]

        raise ValueError(
            f'{self}.outlierRstMax({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def profile(self, *cols: str, **kwargs: Any) -> Namespace:
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
            cols = self.contentCols

        asDict = kwargs.pop('asDict', False)

        if len(cols) > 1:
            return Namespace(**{col: self.profile(col, **kwargs)
                                for col in cols})

        col = cols[0]

        verbose = (True
                   if debug.ON
                   else kwargs.get('verbose'))

        if verbose:
            msg = f'Profiling Column "{col}"...'
            self.stdOutLogger.info(msg)
            tic = time.time()

        colType = self.type(col)
        profile = Namespace(type=colType)

        # non-NULL Proportions
        profile.nonNullProportion = \
            self.nonNullProportion(
                col,
                verbose=verbose > 1)

        if self.suffNonNull(col) or \
                (not kwargs.get('skipIfInsuffNonNull', False)):
            # profile categorical column
            if kwargs.get('profileCat', True) and is_possible_cat(colType):
                profile.distinctProportions = \
                    self.distinct(
                        col,
                        count=True,
                        verbose=verbose > 1)

            # profile numerical column
            if kwargs.get('profileNum', True) and is_num(colType):
                outlierTailProportion = self._outlierTailProportion[col]

                quantilesOfInterest = \
                    Series(
                        index=(0,
                               outlierTailProportion,
                               .5,
                               1 - outlierTailProportion,
                               1))
                quantileProbsToQuery = []

                sampleMin = self._cache.sampleMin.get(col)
                if sampleMin:
                    quantilesOfInterest[0] = sampleMin
                    toCacheSampleMin = False
                else:
                    quantileProbsToQuery += [0.]
                    toCacheSampleMin = True

                outlierRstMin = self._cache.outlierRstMin.get(col)
                if outlierRstMin:
                    quantilesOfInterest[outlierTailProportion] = \
                        outlierRstMin
                    toCacheOutlierRstMin = False
                else:
                    quantileProbsToQuery += [outlierTailProportion]
                    toCacheOutlierRstMin = True

                sampleMedian = self._cache.sampleMedian.get(col)
                if sampleMedian:
                    quantilesOfInterest[.5] = sampleMedian
                    toCacheSampleMedian = False
                else:
                    quantileProbsToQuery += [.5]
                    toCacheSampleMedian = True

                outlierRstMax = self._cache.outlierRstMax.get(col)
                if outlierRstMax:
                    quantilesOfInterest[1 - outlierTailProportion] = \
                        outlierRstMax
                    toCacheOutlierRstMax = False
                else:
                    quantileProbsToQuery += [1 - outlierTailProportion]
                    toCacheOutlierRstMax = True

                sampleMax = self._cache.sampleMax.get(col)
                if sampleMax:
                    quantilesOfInterest[1] = sampleMax
                    toCacheSampleMax = False
                else:
                    quantileProbsToQuery += [1.]
                    toCacheSampleMax = True

                series = self.reprSample[col]

                if quantileProbsToQuery:
                    quantilesOfInterest[
                        isnan(quantilesOfInterest)] = \
                        series.quantile(
                            q=quantileProbsToQuery,
                            interpolation='linear')

                (sampleMin, outlierRstMin,
                 sampleMedian,
                 outlierRstMax, sampleMax) = quantilesOfInterest

                if toCacheSampleMin:
                    self._cache.sampleMin[col] = sampleMin

                if toCacheOutlierRstMin:
                    if (outlierRstMin == sampleMin) and \
                            (outlierRstMin < sampleMedian):
                        outlierRstMin = \
                            series.loc[series > sampleMin] \
                            .min(axis='index', skipna=True, level=None)
                    self._cache.outlierRstMin[col] = outlierRstMin

                if toCacheSampleMedian:
                    self._cache.sampleMedian[col] = sampleMedian

                if toCacheOutlierRstMax:
                    if (outlierRstMax == sampleMax) and \
                            (outlierRstMax > sampleMedian):
                        outlierRstMax = (
                            series
                            .loc[series < sampleMax]
                            .max(axis='index', skipna=True, level=None))
                    self._cache.outlierRstMax[col] = outlierRstMax

                if toCacheSampleMax:
                    self._cache.sampleMax[col] = sampleMax

                profile.sampleRange = sampleMin, sampleMax
                profile.outlierRstRange = outlierRstMin, outlierRstMax

                profile.sampleMean = \
                    self.sampleStat(
                        col,
                        stat='mean',
                        verbose=verbose)

                profile.outlierRstMean = \
                    self._cache.outlierRstMean.get(
                        col,
                        self.outlierRstStat(
                            col,
                            stat='mean',
                            verbose=verbose))

                profile.outlierRstMedian = \
                    self._cache.outlierRstMedian.get(
                        col,
                        self.outlierRstStat(
                            col,
                            stat='median',
                            verbose=verbose))

        if verbose:
            toc = time.time()
            self.stdOutLogger.info(
                msg + f' done!   <{toc - tic:,.1f} s>')

        return (Namespace(**{col: profile})
                if asDict
                else profile)

    # =========
    # DATA PREP
    # ---------
    # fillna
    # prep

    def fillna(self, *cols: str, **kwargs: Any):
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Fill/interpolate ``NULL``/``NaN`` values.

        Return:
            ``ADF`` with ``NULL``/``NaN`` values filled/interpolated

        Args:
            *args (str): names of column(s) to fill/interpolate

            **kwargs:

                - **method** *(str)*: one of the following methods to fill
                    ``NULL`` values in **numerical** columns,
                    or *dict* of such method specifications by column name

                    - ``avg``/``mean`` (default)
                    - ``min``
                    - ``max``
                    - ``avg_before``/``mean_before``
                    - ``min_before``
                    - ``max_before``
                    - ``avg_after``/``mean_after``
                    - ``min_after``
                    - ``max_after``
                    - ``linear`` (**TO-DO**)
                    - ``before`` (**TO-DO**)
                    - ``after`` (**TO-DO**)
                    - ``None`` (do nothing)

                    (*NOTE:* for an ``ADF`` with a ``.tCol`` set,
                     ``NumPy/Pandas NaN`` values cannot be filled;
                     it is best that such *Python* values be cleaned up
                     before they get into Spark)

                - **value**: single value, or *dict* of values by column name,
                    to use if ``method`` is ``None`` or not applicable

                - **outlierTails** *(str or dict of str, default = 'both')*:
                specification of in which distribution tail
                (``None``, ``lower``, ``upper`` and ``both`` (default))
                of each numerical column out-lying values may exist

                - **fillOutliers**
                *(bool or list of column names, default = False)*:
                whether to treat detected out-lying values as ``NULL``
                values to be replaced in the same way

                - **loadPath** *(str)*:
                path to load existing ``NULL``-filling data transformations

                - **savePath** *(str)*: path to save new
                ``NULL``-filling data transformations
        """
        _TS_FILL_METHODS = (
            'avg_partition', 'mean_partition',
            'min_partition', 'max_partition',
            'avg_before', 'mean_before', 'min_before', 'max_before',
            'avg_after', 'mean_after', 'min_after', 'max_after')

        if self.hasTS:
            _TS_OPPOSITE_METHODS = \
                Namespace(
                    avg='avg',
                    mean='mean',
                    min='max',
                    max='min')

            _TS_WINDOW_NAMES = \
                Namespace(
                    partition='partitionByI',
                    before='partitionByI_orderByT_before',
                    after='partitionByI_orderByT_after')

            _TS_OPPOSITE_WINDOW_NAMES = \
                Namespace(
                    partition='partition',
                    before='after',
                    after='before')

            _TS_WINDOW_DEFS = \
                Namespace(
                    partition=   # noqa: E251
                    (f'{_TS_WINDOW_NAMES.partition} AS '
                     f'(PARTITION BY {self._iCol}, __tChunk__)'),

                    before=   # noqa: E251
                    (f'{_TS_WINDOW_NAMES.before} AS '
                     f'(PARTITION BY {self._iCol}, __tChunk__ '
                     f'ORDER BY {self._T_ORD_COL} '
                     'ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)'),

                    after=   # noqa: E251
                    (f'{_TS_WINDOW_NAMES.after} AS '
                     f'(PARTITION BY {self._iCol}, __tChunk__ '
                     f'ORDER BY {self._T_ORD_COL} '
                     'ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)'))

        returnDetails = kwargs.pop('returnDetails', False)
        returnSQLStatement = kwargs.pop('returnSQLStatement', False)
        loadPath = kwargs.pop('loadPath', None)
        savePath = kwargs.pop('savePath', None)

        verbose = kwargs.pop('verbose', False)
        if debug.ON:
            verbose = True

        if loadPath:   # pylint: disable=too-many-nested-blocks
            if verbose:
                message = ('Loading NULL-Filling SQL Statement '
                           f'from Path "{loadPath}"...')
                self.stdOutLogger.info(message)
                tic = time.time()

            with open(os.path.join(loadPath,
                                   self._NULL_FILL_SQL_STATEMENT_FILE_NAME),
                      mode='r',
                      encoding='urf-8') as f:
                sqlStatement = json.load(f)

            details = None

        else:
            value = kwargs.pop('value', None)

            method = kwargs.pop(
                'method',
                'mean' if value is None else None)

            cols = set(cols)

            if isinstance(method, dict):
                cols.update(method)

            if isinstance(value, dict):
                cols.update(value)

            if not cols:
                cols = set(self.contentCols)

            cols.difference_update(self.indexCols)

            nulls = kwargs.pop('nulls', {})

            for col in cols:
                if col in nulls:
                    colNulls = nulls[col]

                    assert (isinstance(colNulls, (list, tuple)) and
                            (len(colNulls) == 2) and
                            ((colNulls[0] is None) or
                             isinstance(colNulls[0], PY_NUM_TYPES)) and
                            ((colNulls[1] is None) or
                             isinstance(colNulls[1], PY_NUM_TYPES)))

                else:
                    nulls[col] = (None, None)

            outlierTails = kwargs.pop('outlierTails', {})
            if isinstance(outlierTails, str):
                outlierTails = {col: outlierTails for col in cols}

            fillOutliers = kwargs.pop('fillOutliers', False)
            fillOutliers = \
                cols \
                if fillOutliers is True \
                else to_iterable(fillOutliers)

            tsWindowDefs = set()
            details = {}

            if verbose:
                message = 'NULL-Filling Columns {}...'.format(
                    ', '.join(f'"{col}"' for col in cols))
                self.stdOutLogger.info(message)
                tic = time.time()

            for col in cols:
                colType = self.type(col)
                colFallBackVal = None

                if is_num(colType):
                    isNum = True

                    colOutlierTails = outlierTails.get(col, 'both')
                    fixLowerTail = colOutlierTails in ('lower', 'both')
                    fixUpperTail = colOutlierTails in ('upper', 'both')

                    methodForCol = \
                        method[col] \
                        if isinstance(method, dict) and (col in method) \
                        else method

                    if methodForCol:
                        methodForCol = methodForCol.lower().split('_')

                        if len(methodForCol) == 2:
                            assert self.hasTS, \
                                ('NULL-Filling Methods '
                                 f"{', '.join(s.upper() for s in _TS_FILL_METHODS)} "
                                 'Not Supported for Non-Time-Series ADFs')

                            methodForCol, window = methodForCol

                        else:
                            methodForCol = methodForCol[0]

                            if self.hasTS:
                                window = None

                        colFallBackVal = \
                            self.outlierRstStat(
                                col,
                                stat=(methodForCol
                                      if (not self.hasTS) or (window is None)
                                      or (window == 'partition')
                                      else 'mean'),
                                outlierTails=colOutlierTails,
                                verbose=verbose > 1)

                    elif isinstance(value, dict):
                        colFallBackVal = value.get(col)
                        if not isinstance(colFallBackVal, PY_NUM_TYPES):
                            colFallBackVal = None

                    elif isinstance(value, PY_NUM_TYPES):
                        colFallBackVal = value

                else:
                    isNum = False

                    if isinstance(value, dict):
                        colFallBackVal = value.get(col)
                        if isinstance(colFallBackVal, PY_NUM_TYPES):
                            colFallBackVal = None

                    elif not isinstance(value, PY_NUM_TYPES):
                        colFallBackVal = value

                if notnull(colFallBackVal):
                    fallbackStrs = \
                        [f"'{colFallBackVal}'"
                         if is_string(colType) and
                         isinstance(colFallBackVal, str)
                         else repr(colFallBackVal)]

                    lowerNull, upperNull = colNulls = nulls[col]

                    if isNum and self.hasTS and window:
                        partitionFallBackStrTemplate = \
                            ("{}(CASE WHEN (STRING({}) = 'NaN'){}{}{}{} "
                             "THEN NULL ELSE {} END) OVER {}")

                        fallbackStrs.insert(
                            0,
                            partitionFallBackStrTemplate.format(
                                methodForCol,
                                col,
                                '' if lowerNull is None
                                else f' OR ({col} <= {lowerNull})',
                                '' if upperNull is None
                                else f' OR ({col} >= {upperNull})',
                                f' OR ({col} < {self.outlierRstMin(col)})'
                                if fixLowerTail
                                else '',
                                f' OR ({col} > {self.outlierRstMax(col)})'
                                if fixUpperTail
                                else '',
                                col,
                                _TS_WINDOW_NAMES[window]))
                        tsWindowDefs.add(_TS_WINDOW_DEFS[window])

                        if window != 'partition':
                            oppositeWindow = _TS_OPPOSITE_WINDOW_NAMES[window]
                            fallbackStrs.insert(
                                1,
                                partitionFallBackStrTemplate.format(
                                    _TS_OPPOSITE_METHODS[methodForCol],
                                    col,
                                    '' if lowerNull is None
                                    else f' OR ({col} <= {lowerNull})',
                                    '' if upperNull is None
                                    else f' OR ({col} >= {upperNull})',
                                    f' OR ({col} < {self.outlierRstMin(col)})'
                                    if fixLowerTail
                                    else '',
                                    f' OR ({col} > {self.outlierRstMax(col)})'
                                    if fixUpperTail
                                    else '',
                                    col,
                                    _TS_WINDOW_NAMES[oppositeWindow]))
                            tsWindowDefs.add(_TS_WINDOW_DEFS[oppositeWindow])

                    details[col] = [
                        self._NULL_FILL_PREFIX + col + self._PREP_SUFFIX,

                        # pylint: disable=line-too-long
                        dict(
                            SQL="COALESCE(CASE WHEN (STRING({0}) = 'NaN'){1}{2}{3}{4} THEN NULL ELSE {0} END, {5})"
                                .format(
                                    col,
                                    '' if lowerNull is None
                                    else f' OR ({col} <= {lowerNull})',
                                    '' if upperNull is None
                                    else f' OR ({col} >= {upperNull})',
                                    f' OR ({col} < {self.outlierRstMin(col)})'
                                    if isNum and (col in fillOutliers)
                                    and fixLowerTail
                                    else '',
                                    f' OR ({col} > {self.outlierRstMax(col)})'
                                    if isNum and (col in fillOutliers)
                                    and fixUpperTail
                                    else '',
                                    ', '.join(fallbackStrs)),

                            Nulls=colNulls,
                            NullFillValue=colFallBackVal)]

            if tsWindowDefs:
                details['__TS_WINDOW_CLAUSE__'] = \
                    _tsWindowClause = \
                    f"WINDOW {', '.join(tsWindowDefs)}"

            else:
                _tsWindowClause = ''

            sqlStatement = \
                'SELECT *, {} FROM __THIS__ {}'.format(
                    ', '.join(
                        '{} AS {}'.format(nullFillDetails['SQL'], nullFillCol)
                        for col, (nullFillCol, nullFillDetails) in
                        details.items()
                        if col != '__TS_WINDOW_CLAUSE__'),
                    _tsWindowClause)

        if savePath and (savePath != loadPath):
            if verbose:
                msg = ('Saving NULL-Filling SQL Statement '
                       f'to Path "{savePath}"...')
                self.stdOutLogger.info(msg)
                _tic = time.time()

            fs.mkdir(
                dir_path=savePath,
                hdfs=False)

            with open(os.path.join(savePath,
                                   self._NULL_FILL_SQL_STATEMENT_FILE_NAME),
                      mode='w',
                      encoding='utf-8') as f:
                json.dump(sqlStatement, f, indent=2)

            if verbose:
                _toc = time.time()
                self.stdOutLogger.info(
                    msg + f' done!   <{_toc - _tic:,.1f} s>')

        arrowADF = \
            self.map(
                _S3ParquetDataFeeder__fillna__pandasDFTransform(
                    nullFillDetails=details),
                inheritNRows=True,
                **kwargs)

        arrowADF._inheritCache(
            self,
            *(() if loadPath else cols))

        arrowADF._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdOutLogger.info(
                message + f' done!   <{((toc - tic) / 60):,.1f} m>')

        return (((arrowADF, details, sqlStatement)
                 if returnSQLStatement
                 else (arrowADF, details))
                if returnDetails
                else arrowADF)

    def prep(self, *cols: str, **kwargs: Any):
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        """Pre-process selected column(s) in standard ways.

        One-hot-encode categorical columns and scale numerical columns.

        Return:
            Standard-pre-processed ``ADF``

        Args:
            *args: column(s) to pre-process

            **kwargs:
                - **forceCat** *(str or list/tuple of str, default = None)*:
                columns to force to be categorical variables

                - **forceNum** *(str or list/tuple of str, default = None)*:
                columns to force to be numerical variables

                - **fill**:
                    - *dict* ( ``method`` = ... *(default: 'mean')*,
                    ``value`` = ... *(default: None)*,
                    ``outlierTails`` = ... *(default: False)*,
                    ``fillOutliers`` = ... *(default: False)*)
                    as per ``.fillna(...)`` method;
                    - *OR* ``None`` to not apply any ``NULL``/``NaN``-filling

                - **scaler** *(str)*: one of the following methods
                to use on numerical columns
                (*ignored* if loading existing
                ``prep`` pipeline from ``loadPath``):

                    - ``standard`` (default)
                    - ``maxabs``
                    - ``minmax``
                    - ``None`` *(do not apply any scaling)*

                - **assembleVec** *(str, default = '__X__')*:
                name of vector column to build from pre-processed features;
                *ignored* if loading existing ``prep`` pipeline from
                ``loadPath``

                - **loadPath** *(str)*: path to
                load existing data transformations

                - **savePath** *(str)*: path to save
                new fitted data transformations
        """
        def sqlStdScl(sqlItem, mean, std):
            return f'(({sqlItem}) - {mean}) / {std}'

        def sqlMaxAbsScl(sqlItem, maxAbs):
            return f'({sqlItem}) / {maxAbs}'

        def sqlMinMaxScl(sqlItem, origMin, origMax, targetMin, targetMax):
            origRange = origMax - origMin
            targetRange = targetMax - targetMin
            return (f'({targetRange} * '
                    f'(({sqlItem}) - ({origMin})) / {origRange})'
                    f' + ({targetMin})')

        nulls = kwargs.pop('nulls', {})

        forceCatIncl = kwargs.pop('forceCatIncl', None)
        forceCatExcl = kwargs.pop('forceCatExcl', None)
        forceCat = kwargs.pop('forceCat', None)
        forceCat = \
            (set()
             if forceCat is None
             else to_iterable(forceCat, iterable_type=set)) \
            .union(
                ()
                if forceCatIncl is None
                else to_iterable(forceCatIncl)) \
            .difference(
                ()
                if forceCatExcl is None
                else to_iterable(forceCatExcl))

        kwargs.pop('oheCat', None)   # *** NOT USED ***
        scaleCat = kwargs.pop('scaleCat', True)

        forceNumIncl = kwargs.pop('forceNumIncl', None)
        forceNumExcl = kwargs.pop('forceNumExcl', None)
        forceNum = kwargs.pop('forceNum', None)
        forceNum = \
            (set()
             if forceNum is None
             else to_iterable(forceNum, iterable_type=set)) \
            .union(
                ()
                if forceNumIncl is None
                else to_iterable(forceNumIncl)) \
            .difference(
                ()
                if forceNumExcl is None
                else to_iterable(forceNumExcl))

        fill = kwargs.pop(
            'fill',
            dict(method='mean',
                 value=None,
                 outlierTails='both',
                 fillOutliers=False))

        assert fill, \
            (f'*** {type(self)}.prep(...) MUST INVOLVE NULL-FILLING '
             f'FOR NUMERIC COLS ***')

        scaler = kwargs.pop('scaler', 'standard')
        if scaler:
            scaler = scaler.lower()

        returnNumPy = kwargs.pop('returnNumPy', False)
        returnOrigToPrepColMaps = kwargs.pop('returnOrigToPrepColMaps', False)
        returnSQLStatement = kwargs.pop('returnSQLStatement', False)

        loadPath = kwargs.pop('loadPath', None)
        savePath = kwargs.pop('savePath', None)

        verbose = kwargs.pop('verbose', False)
        if debug.ON:
            verbose = True

        if loadPath:   # pylint: disable=too-many-nested-blocks
            if verbose:
                message = ('Loading & Applying Data Transformations '
                           f'from Path "{loadPath}"...')
                self.stdOutLogger.info(message)
                tic = time.time()

            if loadPath in self._PREP_CACHE:
                prepCache = self._PREP_CACHE[loadPath]

                catOrigToPrepColMap = prepCache.catOrigToPrepColMap
                numOrigToPrepColMap = prepCache.numOrigToPrepColMap
                defaultVecCols = prepCache.defaultVecCols

                sqlStatement = prepCache.sqlStatement
                # sqlTransformer = prepCache.sqlTransformer

                # catOHETransformer = prepCache.catOHETransformer
                # pipelineModelWithoutVectors = \
                #     prepCache.pipelineModelWithoutVectors

            else:
                if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                    localDirExists = os.path.isdir(loadPath)

                    hdfsDirExists = ...   # TODO   # pylint: disable=fixme

                    if localDirExists and (not hdfsDirExists):
                        fs.put(
                            from_local=loadPath,
                            to_hdfs=loadPath,
                            is_dir=True,
                            _mv=False)

                    elif hdfsDirExists and (not localDirExists):
                        fs.get(
                            from_hdfs=loadPath,
                            to_local=loadPath,
                            is_dir=True,
                            overwrite=True, _mv=False,
                            must_succeed=True,
                            _on_linux_cluster_with_hdfs=True)

                with open(
                        os.path.join(
                            loadPath,
                            self._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME),
                        mode='r',
                        encoding='utf-8') as f:
                    catOrigToPrepColMap = json.load(f)

                with open(
                        os.path.join(
                            loadPath,
                            self._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME),
                        mode='r',
                        encoding='utf-8') as f:
                    numOrigToPrepColMap = json.load(f)

                defaultVecCols = \
                    [catOrigToPrepColMap[catCol][0]
                     for catCol in sorted(set(catOrigToPrepColMap)
                                          .difference(('__OHE__',
                                                       '__SCALE__')))] + \
                    [numOrigToPrepColMap[numCol][0]
                     for numCol in sorted(set(numOrigToPrepColMap)
                                          .difference(('__TS_WINDOW_CLAUSE__',
                                                       '__SCALER__')))]

                with open(os.path.join(loadPath,
                                       self._PREP_SQL_STATEMENT_FILE_NAME),
                          mode='r',
                          encoding='utf-8') as f:
                    sqlStatement = json.load(f)

                self._PREP_CACHE[loadPath] = \
                    Namespace(
                        catOrigToPrepColMap=catOrigToPrepColMap,
                        numOrigToPrepColMap=numOrigToPrepColMap,
                        defaultVecCols=defaultVecCols,

                        sqlStatement=sqlStatement,
                        sqlTransformer=None,

                        catOHETransformer=None,
                        pipelineModelWithoutVectors=None)

        else:
            if cols:
                cols = set(cols)

                cols = cols.intersection(self.possibleFeatureTAuxCols).union(
                    possibleFeatureContentCol
                    for possibleFeatureContentCol in
                    cols.intersection(self.possibleFeatureContentCols)
                    if self.suffNonNull(possibleFeatureContentCol))

            else:
                cols = self.possibleFeatureTAuxCols + \
                    tuple(possibleFeatureContentCol
                          for possibleFeatureContentCol
                          in self.possibleFeatureContentCols
                          if self.suffNonNull(possibleFeatureContentCol))

            if cols:
                profile = \
                    self.profile(
                        *cols,
                        profileCat=True,
                        profileNum=False,   # or bool(fill) or bool(scaler)?
                        skipIfInsuffNonNull=True,
                        asDict=True,
                        verbose=verbose)

            else:
                return self.copy()

            cols = {col for col in cols
                    if self.suffNonNull(col) and
                    (len(profile[col].distinctProportions.loc[
                        # (profile[col].distinctProportions.index != '') &
                        # FutureWarning:
                        # elementwise comparison failed;
                        # returning scalar instead,
                        # but in the future will perform
                        # elementwise comparison
                        notnull(
                            profile[col].distinctProportions.index)]) > 1
                     )}

            if not cols:
                return self.copy()

            catCols = [
                col
                for col in (cols
                            .intersection(self.possibleCatCols)
                            .difference(forceNum))
                if (col in forceCat) or
                (profile[col].distinctProportions
                 .iloc[:self._maxNCats[col]].sum()
                 >= self._minProportionByMaxNCats[col])]

            numCols = [col
                       for col in cols.difference(catCols)
                       if self.typeIsNum(col)]

            cols = catCols + numCols

            if verbose:
                message = \
                    'Prepping Columns {}...'.format(
                        ', '.join(f'"{col}"' for col in cols))
                self.stdOutLogger.info(message)
                tic = time.time()

            prepSqlItems = {}

            catOrigToPrepColMap = \
                dict(__OHE__=False,
                     __SCALE__=scaleCat)

            if catCols:
                if verbose:
                    msg = ('Transforming Categorical Features ' +
                           ', '.join(f'"{catCol}"' for catCol in catCols) +
                           '...')
                    self.stdOutLogger.info(msg)
                    _tic = time.time()

                catIdxCols = []

                if scaleCat:
                    catScaledIdxCols = []

                for catCol in catCols:
                    catIdxCol = (self._CAT_IDX_PREFIX +
                                 catCol +
                                 self._PREP_SUFFIX)

                    catColType = self.type(catCol)

                    if is_boolean(catColType):
                        cats = [0, 1]

                        nCats = 2

                        catIdxSqlItem = \
                            f'CASE WHEN {catCol} IS NULL THEN 2 \
                                   WHEN {catCol} THEN 1 \
                                   ELSE 0 END'

                    else:
                        isStr = is_string(catColType)

                        cats = [
                            cat
                            for cat in
                            (profile[catCol].distinctProportions.index
                             if catCol in forceCat
                             else (profile[catCol].distinctProportions
                                   .index[:self._maxNCats[catCol]]))
                            if notnull(cat) and
                            ((cat != '') if isStr else isfinite(cat))]

                        nCats = len(cats)

                        catIdxSqlItem = \
                            'CASE {} ELSE {} END'.format(
                                ' '.join('WHEN {} THEN {}'.format(
                                         "{} = '{}'".format(
                                             catCol,
                                             cat.replace("'", "''")
                                             .replace('"', '""'))
                                         if isStr
                                         else f'ABS({catCol} - {cat}) < 1e-9',
                                         i)
                                         for i, cat in enumerate(cats)),
                                nCats)

                    if scaleCat:
                        catPrepCol = (self._MIN_MAX_SCL_PREFIX +
                                      self._CAT_IDX_PREFIX +
                                      catCol +
                                      self._PREP_SUFFIX)
                        catScaledIdxCols.append(catPrepCol)

                        prepSqlItems[catPrepCol] = \
                            sqlMinMaxScl(
                                sqlItem=catIdxSqlItem,
                                origMin=0, origMax=nCats,
                                targetMin=-1, targetMax=1)

                    else:
                        catIdxCols.append(catIdxCol)

                        prepSqlItems[catIdxCol] = catIdxSqlItem

                        catPrepCol = catIdxCol

                    catOrigToPrepColMap[catCol] = \
                        [catPrepCol,

                         dict(Cats=cats,
                              NCats=nCats)]

                if verbose:
                    _toc = time.time()
                    self.stdOutLogger.info(
                        msg + f' done!   <{_toc - tic:,.1f} s>')

            numOrigToPrepColMap = \
                dict(__SCALER__=scaler)

            if numCols:
                numScaledCols = []

                if verbose:
                    msg = 'Transforming Numerical Features {}...'.format(
                        ', '.join(f'"{numCol}"' for numCol in numCols))
                    self.stdOutLogger.info(msg)
                    _tic = time.time()

                outlierTails = fill.get('outlierTails', {})
                if isinstance(outlierTails, str):
                    outlierTails = \
                        {col: outlierTails
                         for col in numCols}

                _, numNullFillDetails = \
                    self.fillna(
                        *numCols,
                        nulls=nulls,
                        method=fill.get('method', 'mean'),
                        value=fill.get('value'),
                        outlierTails=outlierTails,
                        fillOutliers=fill.get('fillOutliers', False),
                        returnDetails=True,
                        verbose=verbose > 1)

                for numCol in numCols:
                    colOutlierTails = outlierTails.get(numCol, 'both')

                    excludeLowerTail = colOutlierTails in ('lower', 'both')
                    colMin = self.outlierRstMin(numCol) \
                        if excludeLowerTail \
                        else self.sampleStat(numCol, stat='min')

                    excludeUpperTail = colOutlierTails in ('upper', 'both')
                    colMax = self.outlierRstMax(numCol) \
                        if excludeUpperTail \
                        else self.sampleStat(numCol, stat='max')

                    if colMin < colMax:
                        numColNullFillDetails = numNullFillDetails[numCol][1]

                        numColSqlItem = numColNullFillDetails['SQL']
                        numColNulls = numColNullFillDetails['Nulls']

                        numColNullFillValue = \
                            numColNullFillDetails['NullFillValue']
                        assert allclose(numColNullFillValue, self.outlierRstStat(numCol))

                        if scaler:
                            if scaler == 'standard':
                                scaledCol = (
                                    self._STD_SCL_PREFIX +
                                    numCol +
                                    self._PREP_SUFFIX)

                                series = self.reprSample[numCol]

                                if colOutlierTails == 'both':
                                    series = series.loc[
                                        series.between(
                                            left=colMin,
                                            right=colMax,
                                            inclusive='both')]

                                elif colOutlierTails == 'lower':
                                    series = series.loc[series > colMin]

                                elif colOutlierTails == 'upper':
                                    series = series.loc[series < colMax]

                                stdDev = float(
                                    series.std(
                                        axis='index',
                                        skipna=True,
                                        level=None,
                                        ddof=1))

                                prepSqlItems[scaledCol] = \
                                    sqlStdScl(
                                        sqlItem=numColSqlItem,
                                        mean=numColNullFillValue,
                                        std=stdDev)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=numColNullFillValue,
                                          Mean=numColNullFillValue,
                                          StdDev=stdDev)]

                            elif scaler == 'maxabs':
                                scaledCol = (self._MAX_ABS_SCL_PREFIX +
                                             numCol +
                                             self._PREP_SUFFIX)

                                maxAbs = float(max(abs(colMin), abs(colMax)))

                                prepSqlItems[scaledCol] = \
                                    sqlMaxAbsScl(
                                        sqlItem=numColSqlItem,
                                        maxAbs=maxAbs)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=numColNullFillValue,
                                          MaxAbs=maxAbs)]

                            elif scaler == 'minmax':
                                scaledCol = (self._MIN_MAX_SCL_PREFIX +
                                             numCol +
                                             self._PREP_SUFFIX)

                                prepSqlItems[scaledCol] = \
                                    sqlMinMaxScl(
                                        sqlItem=numColSqlItem,
                                        origMin=colMin, origMax=colMax,
                                        targetMin=-1, targetMax=1)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=numColNullFillValue,
                                          OrigMin=colMin, OrigMax=colMax,
                                          TargetMin=-1, TargetMax=1)]

                            else:
                                raise ValueError(
                                    '*** Scaler must be one of '
                                    '"standard", "maxabs", "minmax" '
                                    'and None ***')

                        else:
                            scaledCol = (self._NULL_FILL_PREFIX +
                                         numCol +
                                         self._PREP_SUFFIX)

                            prepSqlItems[scaledCol] = numColSqlItem

                            numOrigToPrepColMap[numCol] = \
                                [scaledCol,

                                 dict(Nulls=numColNulls,
                                      NullFillValue=numColNullFillValue)]

                        numScaledCols.append(scaledCol)

                if verbose:
                    _toc = time.time()
                    self.stdOutLogger.info(
                        msg + f' done!   <{_toc - _tic:,.1f} s>')

            defaultVecCols = \
                [catOrigToPrepColMap[catCol][0]
                 for catCol in sorted(set(catOrigToPrepColMap)
                                      .difference(('__OHE__',
                                                   '__SCALE__')))] + \
                [numOrigToPrepColMap[numCol][0]
                 for numCol in sorted(set(numOrigToPrepColMap)
                                      .difference(('__TS_WINDOW_CLAUSE__',
                                                   '__SCALER__')))]

            sqlStatement = \
                'SELECT *, {} FROM __THIS__ {}'.format(
                    ', '.join(f'{sqlItem} AS {prepCol}'
                              for prepCol, sqlItem in prepSqlItems.items()),
                    numNullFillDetails.get('__TS_WINDOW_CLAUSE__', ''))

        if savePath and (savePath != loadPath):
            if verbose:
                msg = ('Saving Data Transformations '
                       f'to Local Path "{savePath}"...')
                self.stdOutLogger.info(msg)
                _tic = time.time()

            fs.mkdir(
                dir_path=savePath,
                hdfs=False)

            with open(os.path.join(savePath,
                                   self._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME),
                      mode='w',
                      encoding='utf-8') as f:
                json.dump(catOrigToPrepColMap, f, indent=2)

            with open(os.path.join(savePath,
                                   self._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME),
                      mode='w',
                      encoding='utf-8') as f:
                json.dump(numOrigToPrepColMap, f, indent=2)

            with open(os.path.join(savePath,
                                   self._PREP_SQL_STATEMENT_FILE_NAME),
                      mode='w',
                      encoding='utf-8') as f:
                json.dump(sqlStatement, f, indent=2)

            if verbose:
                _toc = time.time()
                self.stdOutLogger.info(
                    msg + f' done!   <{_toc - _tic:,.1f} s>')

            self._PREP_CACHE[savePath] = \
                Namespace(
                    catOrigToPrepColMap=catOrigToPrepColMap,
                    numOrigToPrepColMap=numOrigToPrepColMap,
                    defaultVecCols=defaultVecCols,

                    sqlStatement=sqlStatement,
                    sqlTransformer=None,

                    catOHETransformer=None,
                    pipelineModelWithoutVectors=None)

        if returnNumPy:
            returnNumPyForCols = \
                sorted(catPrepColDetails[0]
                       for catCol, catPrepColDetails in
                       catOrigToPrepColMap.items()
                       if (catCol not in ('__OHE__', '__SCALE__')) and
                       isinstance(catPrepColDetails, list) and
                       (len(catPrepColDetails) == 2)) + \
                sorted(numPrepColDetails[0]
                       for numCol, numPrepColDetails in
                       numOrigToPrepColMap.items()
                       if (numCol not in ('__TS_WINDOW_CLAUSE__',
                                          '__SCALER__')) and
                       isinstance(numPrepColDetails, list) and
                       (len(numPrepColDetails) == 2))

        else:
            colsToKeep = \
                self.columns + \
                (([catPrepColDetails[0]
                   for catCol, catPrepColDetails in catOrigToPrepColMap.items()
                   if (catCol not in ('__OHE__', '__SCALE__')) and
                   isinstance(catPrepColDetails, list) and
                   (len(catPrepColDetails) == 2)] +
                  [numPrepColDetails[0]
                   for numCol, numPrepColDetails in numOrigToPrepColMap.items()
                   if (numCol not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and
                   isinstance(numPrepColDetails, list) and
                   (len(numPrepColDetails) == 2)])
                 if loadPath
                 else (((catScaledIdxCols
                         if scaleCat
                         else catIdxCols)
                        if catCols
                        else []) +
                       (numScaledCols
                        if numCols
                        else [])))

        missingCatCols = \
            set(catOrigToPrepColMap) \
            .difference(
                self.columns +
                ['__OHE__', '__SCALE__'])

        missingNumCols = \
            set(numOrigToPrepColMap) \
            .difference(
                self.columns +
                ['__TS_WINDOW_CLAUSE__', '__SCALER__'])

        missingCols = missingCatCols | missingNumCols

        addCols = {}

        if missingCols:
            if debug.ON:
                self.stdOutLogger.debug(
                    msg=f'*** FILLING MISSING COLS {missingCols} ***')

            for missingCol in missingCols:
                addCols[missingCol] = \
                    nan \
                    if missingCol in missingCatCols \
                    else numOrigToPrepColMap[missingCol][1]['NullFillValue']

                if not returnNumPy:
                    colsToKeep.append(missingCol)

        arrowADF = \
            self.map(
                _S3ParquetDataFeeder__prep__pandasDFTransform(
                    addCols=addCols,
                    typeStrs={
                        catCol: str(self.type(catCol))
                        for catCol in (set(catOrigToPrepColMap)
                                       .difference(('__OHE__', '__SCALE__')))},
                    catOrigToPrepColMap=catOrigToPrepColMap,
                    numOrigToPrepColMap=numOrigToPrepColMap,
                    returnNumPyForCols=(returnNumPyForCols
                                        if returnNumPy
                                        else None)),
                inheritNRows=True,
                **kwargs)

        if not returnNumPy:
            arrowADF = arrowADF[colsToKeep]

            arrowADF._inheritCache(
                self,
                *(() if loadPath else colsToKeep))

            arrowADF._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdOutLogger.info(
                message + f' done!   <{((toc - tic) / 60):,.1f} m>')

        return (((arrowADF, catOrigToPrepColMap, numOrigToPrepColMap,
                  sqlStatement)
                 if returnSQLStatement
                 else (arrowADF, catOrigToPrepColMap, numOrigToPrepColMap))
                if returnOrigToPrepColMaps
                else arrowADF)

    # ============
    # MISC / OTHER
    # ------------
    # split

    def split(self, *weights: float, **kwargs: Any) \
            -> Union[S3ParquetDataFeeder, List[S3ParquetDataFeeder]]:
        """Split into multiple S3 Parquet Data Feeders by weight."""
        if (not weights) or weights == (1,):
            return self

        nWeights = len(weights)
        cumuWeights = cumsum(weights) / sum(weights)

        nPieces = self.nPieces

        piecePaths = list(self.piecePaths)
        random.shuffle(piecePaths)

        cumuIndices = \
            [0] + \
            [int(round(cumuWeights[i] * nPieces))
                for i in range(nWeights)]

        return [self._subset(*piecePaths[cumuIndices[i]:cumuIndices[i + 1]],
                             **kwargs)
                for i in range(nWeights)]
