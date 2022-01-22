from argparse import Namespace as _Namespace
import datetime
from functools import lru_cache
import json
import math
import os
import random
import re
import sys
import tempfile
import time
from typing import Optional
from urllib.parse import urlparse
import uuid

import numpy
import pandas
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from tqdm import tqdm

from pyarrow.dataset import dataset
from pyarrow.fs import S3FileSystem
from pyarrow.parquet import read_metadata, read_schema, read_table

from h1st_util import DefaultDict, fs, Namespace
from h1st_util.aws import s3
from h1st_util.date_time import gen_aux_cols, DATE_COL
from h1st_util.iterables import to_iterable
from h1st_util.types.arrow import (
    _ARROW_INT_TYPE, _ARROW_DOUBLE_TYPE, _ARROW_STR_TYPE, _ARROW_DATE_TYPE,
    is_binary, is_boolean, is_complex, is_num, is_possible_cat, is_string)
from h1st_util.types.numpy_pandas import NUMPY_FLOAT_TYPES, NUMPY_INT_TYPES, PY_NUM_TYPES   # noqa: E501
from h1st_util.types.spark_sql import _STR_TYPE
import h1st_util.debug

from ._abstract import AbstractDataHandler

if sys.version_info >= (3, 9):
    from collections.abc import Collection, Sequence
else:
    from typing import Collection, Sequence


_NUM_CLASSES = int, float


# pylint: disable=consider-using-f-string,line-too-long,too-many-lines


class AbstractS3ParquetDataHandler(AbstractDataHandler):
    # pylint: disable=abstract-method

    _SCHEMA_MIN_N_PIECES = 10
    _REPR_SAMPLE_MIN_N_PIECES = 100

    @property
    def reprSampleMinNPieces(self):
        return self._reprSampleMinNPieces

    @reprSampleMinNPieces.setter
    def reprSampleMinNPieces(self, reprSampleMinNPieces):
        if (reprSampleMinNPieces <= self.nPieces) and \
                (reprSampleMinNPieces != self._reprSampleMinNPieces):
            self._reprSampleMinNPieces = reprSampleMinNPieces

    @reprSampleMinNPieces.deleter
    def reprSampleMinNPieces(self):
        self._reprSampleMinNPieces = min(self._REPR_SAMPLE_MIN_N_PIECES,
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

        return pandas.Series(index=pandasDF.index, name=self.col)


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
                    (AbstractDataHandler._NULL_FILL_PREFIX +   # noqa: W504
                     col +   # noqa: W504
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
                    AbstractDataHandler._NULL_FILL_PREFIX +   # noqa: W504
                    numCol +   # noqa: W504
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
                numpy.array(
                    [numPrepDetails['Mean']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature relative scaling of the data
            self.numScaler.scale_ = \
                numpy.array(
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
                numpy.array(
                    [numPrepDetails['MaxAbs']
                     for numPrepDetails in self.numPrepDetails])

        elif self.numScaler == 'minmax':
            self.numScaler = \
                MinMaxScaler(
                    feature_range=(-1, 1),
                    copy=True)

            # per-feature minimum seen in the data
            self.numScaler.data_min_ = \
                numpy.array(
                    [numPrepDetails['OrigMin']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature maximum seen in the data
            self.numScaler.data_max_ = \
                numpy.array(
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
                         for i, cat in enumerate(cats)) +   # noqa: W504
                     ((~s.isin(cats)) * nCats)) \
                    if self.typeStrs[catCol] == _STR_TYPE \
                    else (sum(((s - cat).abs().between(left=0,
                                                       right=_FLOAT_ABS_TOL,
                                                       inclusive='both') * i)
                              for i, cat in enumerate(cats)) +   # noqa: W504
                          ((1 -   # noqa: W504
                            sum((s - cat).abs().between(left=0,
                                                        right=_FLOAT_ABS_TOL,
                                                        inclusive='both')
                                for cat in cats)) *   # noqa: W504
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
            return (numpy.hstack(
                    (pandasDF[self.catPrepCols].values,
                     self.numScaler.transform(
                         X=pandasDF[self.numNullFillCols])))
                    if self.numScaler
                    else pandasDF[self.returnNumPyForCols].values)

        if self.numScaler:
            pandasDF[self.numPrepCols] = \
                pandas.DataFrame(
                    data=self.numScaler.transform(
                        X=pandasDF[self.numNullFillCols]))
            # ^^^ SettingWithCopyWarning (?)
            # A value is trying to be set
            # on a copy of a slice from a DataFrame.
            # Try using .loc[row_indexer,col_indexer] = value instead

        return pandasDF


_PIECE_LOCAL_CACHE_PATHS = {}


class _S3ParquetDataFeeder__pieceArrowTableFunc:
    def __init__(self,
                 aws_access_key_id=None, aws_secret_access_key=None,
                 nThreads=1):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        self.nThreads = nThreads

    # @lru_cache(maxsize=68)
    # # *** too memory-intensive esp. when used in multi-proc ***
    def __call__(self, piecePath):
        if piecePath.startswith('s3'):
            # pylint: disable=global-variable-not-assigned
            global _PIECE_LOCAL_CACHE_PATHS

            if piecePath in _PIECE_LOCAL_CACHE_PATHS:
                path = _PIECE_LOCAL_CACHE_PATHS[piecePath]

            else:
                parsedURL = \
                    urlparse(
                        url=piecePath,
                        scheme='',
                        allow_fragments=True)

                _PIECE_LOCAL_CACHE_PATHS[piecePath] = path = \
                    os.path.join(
                        AbstractDataHandler._TMP_DIR_PATH,
                        parsedURL.netloc,
                        parsedURL.path[1:])

                _dir_path = os.path.dirname(path)

                fs.mkdir(dir=_dir_path,
                         hdfs=False)

                while not os.path.isdir(_dir_path):
                    time.sleep(1)

                (s3.client(
                    access_key_id=self.aws_access_key_id,
                    secret_access_key=self.aws_secret_access_key)
                 .download_file(
                    Bucket=parsedURL.netloc,
                    Key=parsedURL.path[1:],
                    Filename=path))

                # make sure AWS S3's asynchronous process has finished
                # downloading a potentially large file
                while not os.path.isfile(path):
                    time.sleep(1)

        else:
            path = piecePath

        return read_table(
            source=path,
            # str, pyarrow.NativeFile, or file-like object
            # If a string passed, can be a single file name or directory name.
            # For file-like objects, only read a single file.
            # Use pyarrow.BufferReader to read a file
            # contained in a bytes or buffer-like object.

            columns=None,
            # list: If not None, only these columns will be read from the file.
            # A column name may be a prefix of a nested field,
            # e.g. 'a' will select 'a.b', 'a.c', and 'a.d.e'.

            use_threads=False,
            # *** will blow up RAM if True and used in multi-processing ***
            # bool, default True
            # Perform multi-threaded column reads.

            metadata=None,
            # FileMetaData – If separately computed

            use_pandas_metadata=False,
            # bool, default False
            # If True and file has custom pandas schema metadata,
            # ensure that index columns are also loaded

            memory_map=False,
            # bool, default False
            # If the source is a file path, use a memory map to read file,
            # which can improve performance in some environments.

            read_dictionary=None,
            # list, default None
            # List of names or column paths (for nested types)
            # to read directly as DictionaryArray.
            # Only supported for BYTE_ARRAY storage.
            # To read a flat column as dictionary-encoded pass the column name.
            # For nested types, you must pass the full column 'path',
            # which could be something like level1.level2.list.item.
            # Refer to the Parquet file's schema to obtain the paths.

            filesystem=None,
            # FileSystem, default None
            # If nothing passed, paths assumed to be found
            # in the local on-disk filesystem.

            filters=None,
            # List[Tuple] or List[List[Tuple]] or None (default))
            # Rows which do not match the filter predicate will be removed
            # from scanned data.
            # Partition keys embedded in a nested directory structure
            # will be exploited
            # to avoid loading files at all if they contain no matching rows.
            # If use_legacy_dataset is True,
            # filters can only reference partition keys
            # and only a hive-style directory structure is supported.
            # When setting use_legacy_dataset to False,
            # also within-file level filtering an
            #  different partitioning schemes are supported.
            # Predicates are expressed in disjunctive normal form (DNF),
            # like [[('x', '=', 0), ...], ...].
            # DNF allows arbitrary boolean logical combinations
            # of single column predicates.
            # The innermost tuples each describe a single column predicate.
            # The list of inner predicates is interpreted
            # as a conjunction (AND),
            # forming a more selective and multiple column predicate.
            # Finally, the most outer list combines these filters
            # as a disjunction (OR).
            # Predicates may also be passed as List[Tuple].
            # This form is interpreted as a single conjunction.
            # To express OR in predicates, one must use the (preferred)
            # List[List[Tuple]] notation.

            buffer_size=0,
            # int, default 0
            # If positive, perform read buffering
            # when deserializing individual column chunks.
            # Otherwise IO calls are unbuffered.

            partitioning='hive',
            # Partitioning or str or list of str, default "hive"
            # The partitioning scheme for a partitioned dataset.
            # The default of 'hive' assumes directory names
            # with key=value pairs like '/year=2009/month=11'.
            # In addition, a scheme like '/2009/11' is also supported,
            # in which case you need to specify the field names
            # or a full schema. See the pyarrow.dataset.partitioning()
            # function for more details.

            use_legacy_dataset=True,
            # bool, default False
            # By default, read_table uses the new Arrow Datasets API
            # since pyarrow 1.0.0.
            # Among other things, this allows to pass filters for all columns
            # and not only the partition keys,
            # enables different partitioning schemes, etc.
            # Set to True to use the legacy behaviour.

            ignore_prefixes=None
            # list, optional
            # Files matching any of these prefixes will be ignored by
            # the discovery process
            # if use_legacy_dataset=False.
            # This is matched to the basename of a path.
            # By default this is ['.', '_'].
            # Note that discovery happens only
            # if a directory is passed as source.
        )


class _S3ParquetDataFeeder__gen:
    def __init__(
            self, args,
            piecePaths,
            aws_access_key_id, aws_secret_access_key,
            partitionKVs,
            iCol, tCol,
            possibleFeatureTAuxCols, contentCols,
            pandasDFTransforms,
            filterConditions,
            n, sampleN, pad,
            anon,
            nThreads):
        def cols_rowFrom_rowTo(x):
            if isinstance(x, str):
                return [x], None, None

            if isinstance(x, (list, tuple)) and x:
                lastItem = x[-1]

                if isinstance(lastItem, str):
                    return x, None, None

                if isinstance(lastItem, int):
                    secondLastItem = x[-2]
                    return ((x[:-1], 0, lastItem)
                            if lastItem >= 0
                            else (x[:-1], lastItem, 0)) \
                        if isinstance(secondLastItem, str) \
                        else (x[:-2], secondLastItem, lastItem)

        self.piecePaths = list(piecePaths)

        self.partitionKVs = partitionKVs

        self.nThreads = nThreads

        self.pieceArrowTableFunc = \
            _S3ParquetDataFeeder__pieceArrowTableFunc(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                nThreads=nThreads)

        self.filterConditions = filterConditions

        if filterConditions and h1st_util.debug.ON:
            print(f'*** FILTER CONDITION: {filterConditions} ***')

        self.n = n
        self.sampleN = sampleN

        self.pad = pad

        self.anon = anon

        self.iCol = iCol
        self.tCol = tCol

        self.hasTS = self.iCol and self.tCol

        self.possibleFeatureTAuxCols = possibleFeatureTAuxCols
        self.contentCols = contentCols

        self.pandasDFTransforms = pandasDFTransforms

        _hasTS = False
        minTOrd = 0

        if args:
            if self.hasTS:
                self.colsLists = []
                self.colsOverTime = []
                self.rowFrom_n_rowTo_tups = []

                for cols, rowFrom, rowTo in map(cols_rowFrom_rowTo, args):
                    self.colsLists.append(cols)

                    if (rowFrom is None) and (rowTo is None):
                        self.colsOverTime.append(False)
                        self.rowFrom_n_rowTo_tups.append(None)

                    else:
                        assert rowFrom < rowTo <= 0

                        self.colsOverTime.append(True)
                        self.rowFrom_n_rowTo_tups.append((rowFrom, rowTo))

                        _hasTS = True

                        if -rowFrom > minTOrd:
                            minTOrd = -rowFrom

            else:
                self.colsLists = list(args)
                nArgs = len(args)
                self.colsOverTime = nArgs * [False]
                self.rowFrom_n_rowTo_tups = nArgs * [None]

        else:
            self.colsLists = [self.possibleFeatureTAuxCols + self.contentCols]
            self.colsOverTime = [False]
            self.rowFrom_n_rowTo_tups = [None]

        if _hasTS:
            self.filterConditions[AbstractDataHandler._T_ORD_COL] = (minTOrd,
                                                                     numpy.inf)

        if (not self.anon) and (self.iCol or self.tCol):
            self.colsLists.insert(
                0,
                ([self.iCol] if self.iCol else []) +   # noqa: W504
                ([self.tCol] if self.tCol else []))
            self.colsOverTime.insert(0, False)
            self.rowFrom_n_rowTo_tups.insert(0, None)

        self.nColsList = [len(cols) for cols in self.colsLists]

    def __call__(self):
        if h1st_util.debug.ON:
            print(f'*** GENERATING BATCHES OF {self.colsLists} ***')

        while True:
            piecePath = random.choice(self.piecePaths)

            chunkPandasDF = \
                random.choice(
                    self.pieceArrowTableFunc(piecePath=piecePath)
                        .to_batches(max_chunksize=self.sampleN)) \
                .to_pandas(
                    categories=None,
                    strings_to_categorical=False,
                    zero_copy_only=False,
                    integer_object_nulls=False,
                    date_as_object=True,
                    use_threads=False,
                    # single thread sufficient to process 1 chunk of data
                    deduplicate_objects=False,
                    ignore_metadata=False)

            if self.partitionKVs:
                for k, v in self.partitionKVs[piecePath].items():
                    chunkPandasDF[k] = v

            else:
                for partitionKV in re.findall('[^/]+=[^/]+/', piecePath):
                    k, v = partitionKV.split('=')
                    k = str(k)   # ensure not Unicode

                    chunkPandasDF[k] = \
                        datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date() \
                        if k == DATE_COL \
                        else v[:-1]

            if self.tCol:
                chunkPandasDF = \
                    gen_aux_cols(
                        df=chunkPandasDF,
                        i_col=self.iCol,
                        t_col=self.tCol)

            for i, pandasDFTransform in enumerate(self.pandasDFTransforms):
                try:
                    chunkPandasDF = pandasDFTransform(chunkPandasDF)

                except Exception as err:
                    print(f'*** "{piecePath}": PANDAS TRANSFORM #{i} ***')

                    # stackoverflow.com/questions/4825234/
                    # exception-traceback-is-hidden-if-not-re-raised-immediately
                    raise err

            if self.filterConditions:
                filterChunkPandasDF = \
                    chunkPandasDF[list(self.filterConditions)]

                rowIndices = \
                    filterChunkPandasDF.loc[
                        sum(   # AVOID INCLUDING EXTREMES EQUALLING MEDIAN
                            (filterChunkPandasDF[filterCol]
                                .between(
                                    left=left,
                                    right=right,
                                    inclusive='neither')
                             if pandas.notnull(left) and pandas.notnull(right)
                             else ((filterChunkPandasDF[filterCol] > left)
                                   if pandas.notnull(left)
                                   else ((filterChunkPandasDF[filterCol]
                                          < right))))   # noqa: W503
                            for filterCol, (left, right) in
                            self.filterConditions.items())
                        == len(self.filterConditions)].index.tolist()   # noqa: E501,W503

            else:
                rowIndices = chunkPandasDF.index.tolist()

            random.shuffle(rowIndices)

            n_batches = int(math.ceil(len(rowIndices) / self.n))

            for i in range(n_batches):
                rowIndicesSubset = rowIndices[(i * self.n):((i + 1) * self.n)]

                arrays = tuple(
                    (numpy.vstack(
                        numpy.expand_dims(
                            numpy.vstack(
                                (numpy.full(
                                    shape=(max((rowFrom_n_rowTo[1] -   # noqa: E501,W504
                                                rowFrom_n_rowTo[0] + 1)
                                               - max(rowIdx +   # noqa: E501,W503,W504
                                                     rowFrom_n_rowTo[1] + 1,
                                                     0),
                                               0),
                                           nCols),
                                    fill_value=self.pad),
                                 chunkPandasDF.loc[
                                     (rowIdx + rowFrom_n_rowTo[0]):
                                     (rowIdx + rowFrom_n_rowTo[1]),
                                     cols].values
                                    # *** NOTE: pandas.DataFrame.loc[i:j, ...]
                                    # is INCLUSIVE OF j ***
                                 )),
                            axis=0)
                        for rowIdx in rowIndicesSubset)
                     if overTime
                     else chunkPandasDF.loc[rowIndicesSubset, cols].values)
                    for cols, nCols, overTime, rowFrom_n_rowTo in
                    zip(self.colsLists, self.nColsList, self.colsOverTime,
                        self.rowFrom_n_rowTo_tups)
                )

                if h1st_util.debug.ON:
                    for array in arrays:
                        nNaNs = numpy.isnan(array).sum()
                        assert not nNaNs, \
                            f'*** {array.shape}: {nNaNs} NaNs ***'

                yield arrays


def random_sample(population: Collection, k: int) -> list:
    return (random.sample(population=population, k=k)
            if len(population) > k
            else list(population))


class S3ParquetDataFeeder(AbstractS3ParquetDataHandler):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    _CACHE = {}

    _PIECE_CACHES = {}

    _T_AUX_COL_ARROW_TYPES = {
        AbstractS3ParquetDataHandler._T_ORD_COL: _ARROW_INT_TYPE,
        AbstractS3ParquetDataHandler._T_DELTA_COL: _ARROW_DOUBLE_TYPE,

        # Half of Year
        AbstractS3ParquetDataHandler._T_HoY_COL: _ARROW_INT_TYPE,

        # Quarter of Year
        AbstractS3ParquetDataHandler._T_QoY_COL: _ARROW_INT_TYPE,

        # Month of Year
        AbstractS3ParquetDataHandler._T_MoY_COL: _ARROW_INT_TYPE,

        # Week of Year
        # AbstractS3ParquetDataHandler._T_WoY_COL: _ARROW_INT_TYPE,

        # Day of Year
        # AbstractS3ParquetDataHandler._T_DoY_COL: _ARROW_INT_TYPE,

        # Part/Proportion/Fraction of Year
        AbstractS3ParquetDataHandler._T_PoY_COL: _ARROW_DOUBLE_TYPE,

        # Quarter of Half-Year
        AbstractS3ParquetDataHandler._T_QoH_COL: _ARROW_INT_TYPE,

        # Month of Half-Year
        AbstractS3ParquetDataHandler._T_MoH_COL: _ARROW_INT_TYPE,

        # Part/Proportion/Fraction of Half-Year
        AbstractS3ParquetDataHandler._T_PoH_COL: _ARROW_DOUBLE_TYPE,

        # Month of Quarter
        AbstractS3ParquetDataHandler._T_MoQ_COL: _ARROW_INT_TYPE,

        # Part/Proportion/Fraction of Quarter
        AbstractS3ParquetDataHandler._T_PoQ_COL: _ARROW_DOUBLE_TYPE,

        # Week of Month
        AbstractS3ParquetDataHandler._T_WoM_COL: _ARROW_INT_TYPE,

        # Day of Month
        AbstractS3ParquetDataHandler._T_DoM_COL: _ARROW_INT_TYPE,

        # Part/Proportion/Fraction of Month
        AbstractS3ParquetDataHandler._T_PoM_COL: _ARROW_DOUBLE_TYPE,

        # Day of Week
        AbstractS3ParquetDataHandler._T_DoW_COL: _ARROW_INT_TYPE,

        # Part/Proportion/Fraction of Week
        AbstractS3ParquetDataHandler._T_PoW_COL: _ARROW_DOUBLE_TYPE,

        # Hour of Day
        AbstractS3ParquetDataHandler._T_HoD_COL: _ARROW_INT_TYPE,

        # Part/Proportion/Fraction of Day
        AbstractS3ParquetDataHandler._T_PoD_COL: _ARROW_DOUBLE_TYPE,
    }

    # default arguments dict
    _DEFAULT_KWARGS = dict(
        iCol=AbstractS3ParquetDataHandler._DEFAULT_I_COL,
        tCol=None,

        reprSampleMinNPieces=   # noqa: E251
        AbstractS3ParquetDataHandler._REPR_SAMPLE_MIN_N_PIECES,
        reprSampleSize=   # noqa: E251
        AbstractS3ParquetDataHandler._DEFAULT_REPR_SAMPLE_SIZE,

        nulls=   # noqa: E251
        DefaultDict((None, None)),
        minNonNullProportion=   # noqa: E251
        DefaultDict(
            AbstractS3ParquetDataHandler._DEFAULT_MIN_NON_NULL_PROPORTION),
        outlierTailProportion=   # noqa: E251
        DefaultDict(
            AbstractS3ParquetDataHandler._DEFAULT_OUTLIER_TAIL_PROPORTION),
        maxNCats=   # noqa: E251
        DefaultDict(
            AbstractS3ParquetDataHandler._DEFAULT_MAX_N_CATS),
        minProportionByMaxNCats=   # noqa: E251
        DefaultDict(
            AbstractS3ParquetDataHandler._DEFAULT_MIN_PROPORTION_BY_MAX_N_CATS)
    )

    # =================
    # METHODS TO CREATE
    # -----------------
    # __init__
    # load

    def __init__(self,
                 path: str,
                 reCache: bool = False,
                 aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None,
                 aws_region: Optional[str] = None,
                 _mappers: Optional[Sequence[callable]] = None,
                 verbose: bool = True,
                 **kwargs):
        # pylint: disable=too-many-arguments,too-many-branches
        # pylint: disable=too-many-locals,too-many-statements

        if verbose or h1st_util.debug.ON:
            logger = self.class_stdout_logger()

        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region

        assert isinstance(path, str) and path.startswith('s3://')
        self.path = path

        if (not reCache) and (path in self._CACHE):
            _cache = self._CACHE[path]

        else:
            self._CACHE[path] = _cache = Namespace()

        if _cache:
            if h1st_util.debug.ON:
                logger.debug(f'*** RETRIEVING CACHE FOR "{path}" ***')

        else:
            self.s3Client = _cache.s3Client = \
                s3.client(access_key_id=aws_access_key_id,
                          secret_access_key=aws_secret_access_key)

            _parsedURL = urlparse(url=path, scheme='', allow_fragments=True)
            _cache.s3Bucket = _parsedURL.netloc
            _cache.pathS3Key = _parsedURL.path[1:]

            _cache.tmpDirS3Key = 'tmp'
            _cache.tmpDirPath = f's3://{_cache.s3Bucket}/{_cache.tmpDirS3Key}'

            if path in self._PIECE_CACHES:
                _cache.nPieces = 1
                _cache.piecePaths = {path}

            else:
                if verbose:
                    msg = f'Loading "{path}" by Arrow...'
                    logger.info(msg)
                    tic = time.time()

                s3.rm(path=path,
                      dir=True,
                      globs='*_$folder$',   # redundant AWS EMR-generated files
                      quiet=True,
                      access_key_id=aws_access_key_id,
                      secret_access_key=aws_secret_access_key,
                      verbose=False)

                _cache._srcArrowDS = \
                    dataset(source=path.replace('s3://', ''),
                            schema=None,
                            format='parquet',
                            filesystem=S3FileSystem(
                                access_key=aws_access_key_id,
                                secret_key=aws_secret_access_key,
                                region=aws_region),
                            partitioning=None,
                            partition_base_dir=None,
                            exclude_invalid_files=None,
                            ignore_prefixes=None)

                if verbose:
                    toc = time.time()
                    logger.info(msg + f' done!   <{toc - tic:,.1f} s>')

                if (_file_paths := _cache._srcArrowDS.files):
                    _cache.piecePaths = {
                        f's3://{file_path}'
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
                    pieceCache = self._PIECE_CACHES[piecePath]

                    if (pieceCache.nRows is None) and \
                            (i < self._SCHEMA_MIN_N_PIECES):
                        pieceCache.localPath = \
                            self.pieceLocalPath(piecePath=piecePath)

                        schema = read_schema(where=pieceCache.localPath)

                        pieceCache.srcColsExclPartitionKVs = schema.names

                        pieceCache.srcColsInclPartitionKVs += schema.names

                        for col in (set(schema.names)
                                    .difference(pieceCache.partitionKVs)):
                            pieceCache.srcTypesExclPartitionKVs[col] = \
                                pieceCache.srcTypesInclPartitionKVs[col] = \
                                schema.field(col).type

                        metadata = read_metadata(
                            where=pieceCache.localPath)
                        pieceCache.nCols = metadata.num_columns
                        pieceCache.nRows = metadata.num_rows

                else:
                    srcColsInclPartitionKVs = []

                    srcTypesExclPartitionKVs = Namespace()
                    srcTypesInclPartitionKVs = Namespace()

                    partitionKVs = {}

                    for partitionKV in re.findall('[^/]+=[^/]+/', piecePath):
                        k, v = partitionKV.split('=')
                        k = str(k)   # ensure not Unicode

                        srcColsInclPartitionKVs.append(k)

                        if k == DATE_COL:
                            srcTypesInclPartitionKVs[k] = _ARROW_DATE_TYPE
                            partitionKVs[k] = \
                                datetime.datetime.strptime(v[:-1],
                                                           '%Y-%m-%d').date()

                        else:
                            srcTypesInclPartitionKVs[k] = _ARROW_STR_TYPE
                            partitionKVs[k] = v[:-1]

                    if i < self._SCHEMA_MIN_N_PIECES:
                        localPath = self.pieceLocalPath(piecePath=piecePath)

                        schema = read_schema(where=localPath)

                        srcColsExclPartitionKVs = schema.names

                        srcColsInclPartitionKVs += schema.names

                        for col in set(schema.names).difference(partitionKVs):
                            srcTypesExclPartitionKVs[col] = \
                                srcTypesInclPartitionKVs[col] = \
                                schema.field(col).type

                        metadata = read_metadata(where=localPath)
                        nCols = metadata.num_columns
                        nRows = metadata.num_rows

                    else:
                        localPath = None

                        srcColsExclPartitionKVs = None

                        nCols = nRows = None

                    self._PIECE_CACHES[piecePath] = \
                        pieceCache = \
                        Namespace(
                            localPath=localPath,
                            partitionKVs=partitionKVs,

                            srcColsExclPartitionKVs=srcColsExclPartitionKVs,
                            srcColsInclPartitionKVs=srcColsInclPartitionKVs,

                            srcTypesExclPartitionKVs=srcTypesExclPartitionKVs,
                            srcTypesInclPartitionKVs=srcTypesInclPartitionKVs,

                            nCols=nCols,
                            nRows=nRows)

                _cache.srcColsInclPartitionKVs.update(
                    pieceCache.srcColsInclPartitionKVs)

                for col, arrowType in \
                        pieceCache.srcTypesInclPartitionKVs.items():
                    if col in _cache.srcTypesInclPartitionKVs:
                        assert arrowType == \
                            _cache.srcTypesInclPartitionKVs[col], \
                            (f'*** {piecePath} COLUMN {col}: '
                             f'DETECTED TYPE {arrowType} != '
                             f'{_cache.srcTypesInclPartitionKVs[col]} ***')

                    else:
                        _cache.srcTypesInclPartitionKVs[col] = arrowType

        self.__dict__.update(_cache)

        self._cachedLocally = False

        self._mappers = [] if _mappers is None else _mappers

        # extract standard keyword arguments
        self._extractStdKwArgs(kwargs, resetToClassDefaults=True, inplace=True)

        # organize time series if applicable
        self._organizeTimeSeries()

        # set profiling settings and create empty profiling cache
        self._emptyCache()

    @classmethod
    def load(cls, path, **kwargs):
        return cls(path=path, **kwargs)

    # ================================
    # "INTERNAL / DON'T TOUCH" METHODS
    # --------------------------------
    # _extractStdKwArgs
    # _organizeTimeSeries
    # _emptyCache
    # _inheritCache
    # _inplace

    # pylint: disable=inconsistent-return-statements
    def _extractStdKwArgs(self,
                          kwargs,
                          resetToClassDefaults=False,
                          inplace=False):
        nameSpace = self \
            if inplace \
            else _Namespace()

        for k, classDefaultV in self._DEFAULT_KWARGS.items():
            _privateK = f'_{k}'

            if not resetToClassDefaults:
                existingInstanceV = getattr(self, _privateK, None)

            v = kwargs.pop(
                k,
                existingInstanceV
                if (not resetToClassDefaults) and existingInstanceV
                else classDefaultV)

            if (k == 'reprSampleMinNPieces') and (v > self.nPieces):
                v = self.nPieces

            setattr(
                nameSpace,
                _privateK   # *** USE _k TO NOT INVOKE @k.setter RIGHT AWAY ***
                if inplace
                else k,
                v)

        if inplace:
            cols = self.srcColsInclPartitionKVs

            if self._iCol not in cols:
                self._iCol = None

            if self._tCol not in cols:
                self._tCol = None

        else:
            return nameSpace

    def _organizeTimeSeries(self):
        self._dCol = DATE_COL \
            if DATE_COL in self.srcColsInclPartitionKVs \
            else None

        self.hasTS = self._iCol and self._tCol

    def _emptyCache(self):
        self._cache = \
            _Namespace(
                prelimReprSamplePiecePaths=None,
                reprSamplePiecePaths=None,
                reprSample=None,

                approxNRows=None,
                nRows=None,

                count={}, distinct={},   # approx.

                nonNullProportion={},   # approx.
                suffNonNullProportionThreshold={},
                suffNonNull={},

                sampleMin={}, sampleMax={}, sampleMean={}, sampleMedian={},
                outlierRstMin={}, outlierRstMax={},
                outlierRstMean={}, outlierRstMedian={},

                colWidth={})

    def _inheritCache(self, arrowDF, *sameCols, **newColToOldColMappings):
        # pylint: disable=protected-access

        if arrowDF._cache.nRows:
            if self._cache.nRows is None:
                self._cache.nRows = arrowDF._cache.nRows
            else:
                assert self._cache.nRows == arrowDF._cache.nRows

        if arrowDF._cache.approxNRows and (self._cache.approxNRows is None):
            self._cache.approxNRows = arrowDF._cache.approxNRows

        commonCols = set(self.columns).intersection(arrowDF.columns)

        if sameCols or newColToOldColMappings:
            for newCol, oldCol in newColToOldColMappings.items():
                assert newCol in self.columns
                assert oldCol in arrowDF.columns

            for sameCol in (commonCols
                            .difference(newColToOldColMappings)
                            .intersection(sameCols)):
                newColToOldColMappings[sameCol] = sameCol

        else:
            newColToOldColMappings = \
                {col: col
                 for col in commonCols}

        for cacheCategory in \
                ('count', 'distinct',
                 'nonNullProportion',
                 'suffNonNullProportionThreshold',
                 'suffNonNull',
                 'sampleMin', 'sampleMax', 'sampleMean', 'sampleMedian',
                 'outlierRstMin', 'outlierRstMax',
                 'outlierRstMean', 'outlierRstMedian',
                 'colWidth'):
            for newCol, oldCol in newColToOldColMappings.items():
                if oldCol in arrowDF._cache.__dict__[cacheCategory]:
                    self._cache.__dict__[cacheCategory][newCol] = \
                        arrowDF._cache.__dict__[cacheCategory][oldCol]

    def _inplace(self, arrowADF):
        # just in case we're taking in multiple inputs
        if isinstance(arrowADF, (tuple, list)):
            arrowADF = arrowADF[0]

        assert isinstance(arrowADF, S3ParquetDataFeeder)

        self.path = arrowADF.path

        self.__dict__.update(self._CACHE[arrowADF.path])

        self._mappers = arrowADF._mappers

        self._cache = arrowADF._cache

    # ========
    # ITERATOR
    # --------
    # __iter__
    # __next__ / next

    def __iter__(self):
        arrowADF = self.copy(inheritCache=True, inheritNRows=True)
        arrowADF.piecePathsToIter = arrowADF.piecePaths.copy()
        return arrowADF

    def __next__(self):
        if self.piecePathsToIter:
            return self.reduce(
                self.piecePathsToIter.pop(),
                verbose=False)

        raise StopIteration

    # ==========
    # IO METHODS
    # ----------
    # save

    def save(self, dir_path, collect=False, verbose=True):
        if dir_path.startswith('s3://'):
            _s3 = True
            _dir_path = tempfile.mkdtemp()

        else:
            _s3 = False
            _dir_path = dir_path
            fs.empty(
                dir=_dir_path,
                hdfs=False)

        if verbose:
            msg = f'Saving to "{_dir_path}"...'
            self.stdout_logger.info(msg)
            tic = time.time()

        if collect:
            pandasDF = self.collect(verbose=verbose)

            # ValueError: parquet must have string column names
            pandasDF.columns = \
                pandasDF.columns.map(str)

            pandasDF.to_parquet(
                fname=os.path.join(_dir_path, '0.snappy.parquet'),
                engine='pyarrow',
                compression='snappy',
                row_group_size=None,
                # version='1.0',
                use_dictionary=True,
                use_deprecated_int96_timestamps=None,
                coerce_timestamps=None,
                flavor='spark')

        else:
            # pylint: disable=consider-using-f-string
            file_name_formatter = \
                '{:0%dd}.snappy.parquet' % len(str(self.nPieces))

            for i, pandasDF in \
                    (tqdm(enumerate(self), total=self.nPieces)
                     if verbose and (self.nPieces > 1)
                     else enumerate(self)):
                # ValueError: parquet must have string column names
                pandasDF.columns = \
                    pandasDF.columns.map(str)

                pandasDF.to_parquet(
                    fname=os.path.join(_dir_path,
                                       file_name_formatter.format(i)),
                    engine='pyarrow',
                    compression='snappy',
                    row_group_size=None,
                    # version='1.0',
                    use_dictionary=True,
                    use_deprecated_int96_timestamps=None,
                    coerce_timestamps=None,
                    flavor='spark')

        if verbose:
            toc = time.time()
            self.stdout_logger.info(
                msg + f'done!   <{((toc - tic) / 60):,.1f} m>')

        if _s3:
            s3.sync(
                from_dir_path=_dir_path,
                to_dir_path=dir_path,
                access_key_id=self.aws_access_key_id,
                secret_access_key=self.aws_secret_access_key,
                delete=True, quiet=True,
                verbose=verbose)

            fs.rm(
                path=_dir_path,
                hdfs=False,
                is_dir=True)

    def copy(self, **kwargs):
        resetMappers = kwargs.pop('resetMappers', False)
        inheritCache = kwargs.pop('inheritCache', not resetMappers)
        inheritNRows = kwargs.pop('inheritNRows', inheritCache)

        arrowADF = \
            S3ParquetDataFeeder(
                path=self.path,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_region=self.aws_region,

                iCol=self._iCol, tCol=self._tCol,

                _mappers=[] if resetMappers else self._mappers,

                reprSampleMinNPieces=self._reprSampleMinNPieces,
                reprSampleSize=self._reprSampleSize,

                minNonNullProportion=self._minNonNullProportion,
                outlierTailProportion=self._outlierTailProportion,
                maxNCats=self._maxNCats,
                minProportionByMaxNCats=self._minProportionByMaxNCats,

                **kwargs)

        if inheritCache:
            arrowADF._inheritCache(self)

        if inheritNRows:
            arrowADF._cache.approxNRows = self._cache.approxNRows
            arrowADF._cache.nRows = self._cache.nRows

        return arrowADF

    # ===============
    # PYTHON REPR/STR
    # ---------------
    # __repr__
    # __short_repr__

    def __repr__(self):
        col_and_type_strs = []

        if self._iCol:
            col_and_type_strs.append(
                f'(iCol) {self._iCol}: {self.type(self._iCol)}')

        if self._dCol:
            col_and_type_strs.append(
                f'(dCol) {self._dCol}: {self.type(self._dCol)}')

        if self._tCol:
            col_and_type_strs.append(
                f'(tCol) {self._tCol}: {self.type(self._tCol)}')

        col_and_type_strs.extend(f'{col}: {self.type(col)}'
                                 for col in self.contentCols)

        return (f'{self.nPieces:,}-piece ' +   # noqa: W504
                (f'{self._cache.nRows:,}-row '
                 if self._cache.nRows
                 else (f'approx-{self._cache.approxNRows:,.0f}-row '
                       if self._cache.approxNRows
                       else '')) +   # noqa: W504
                type(self).__name__ +   # noqa: W504
                (f'[{self.path} + {len(self._mappers):,} transform(s)]'
                 f"[{', '.join(col_and_type_strs)}]"))

    @property
    def __short_repr__(self):
        cols_desc_str = []

        if self._iCol:
            cols_desc_str.append(f'iCol: {self._iCol}')

        if self._dCol:
            cols_desc_str.append(f'dCol: {self._dCol}')

        if self._tCol:
            cols_desc_str.append(f'tCol: {self._tCol}')

        cols_desc_str.append(f'{len(self.contentCols)} content col(s)')

        return (f'{self.nPieces:,}-piece ' +   # noqa: W504
                (f'{self._cache.nRows:,}-row '
                 if self._cache.nRows
                 else (f'approx-{self._cache.approxNRows:,.0f}-row '
                       if self._cache.approxNRows
                       else '')) +   # noqa: W504
                type(self).__name__ +   # noqa: W504
                (f'[{self.path} + {len(self._mappers):,} transform(s)]'
                 f"[{', '.join(cols_desc_str)}]"))

    # ===============
    # CACHING METHODS
    # ---------------
    # cacheLocally
    # pieceLocalPath

    def cacheLocally(self):
        if not self._cachedLocally:
            parsedURL = urlparse(url=self.path,
                                 scheme='',
                                 allow_fragments=True)

            localPath = os.path.join(self._TMP_DIR_PATH,
                                     parsedURL.netloc,
                                     parsedURL.path[1:])

            s3.sync(from_dir_path=self.path,
                    to_dir_path=localPath,
                    access_key_id=self.aws_access_key_id,
                    secret_access_key=self.aws_secret_access_key,
                    delete=True, quiet=True,
                    verbose=True)

            for piecePath in self.piecePaths:
                self._PIECE_CACHES[piecePath].localPath = \
                    piecePath.replace(self.path, localPath)

            self._cachedLocally = True

    def pieceLocalPath(self, piecePath):
        if (piecePath in self._PIECE_CACHES) and \
                self._PIECE_CACHES[piecePath].localPath:
            return self._PIECE_CACHES[piecePath].localPath

        parsedURL = urlparse(url=piecePath, scheme='', allow_fragments=True)

        localPath = os.path.join(self._TMP_DIR_PATH,
                                 parsedURL.netloc,
                                 parsedURL.path[1:])

        localDirPath = os.path.dirname(localPath)
        fs.mkdir(dir=localDirPath, hdfs=False)
        # make sure the dir has been created
        while not os.path.isdir(localDirPath):
            time.sleep(1)

        self.s3Client.download_file(Bucket=parsedURL.netloc,
                                    Key=parsedURL.path[1:],
                                    Filename=localPath)
        # make sure AWS S3's asynchronous process has finished
        # downloading a potentially large file
        while not os.path.isfile(localPath):
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

    def map(self, mapper=None, **kwargs):
        if mapper is None:
            mapper = []

        inheritCache = kwargs.pop('inheritCache', False)
        inheritNRows = kwargs.pop('inheritNRows', inheritCache)

        additionalMappers = \
            mapper \
            if isinstance(mapper, list) \
            else [mapper]

        arrowADF = \
            S3ParquetDataFeeder(
                path=self.path,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_region=self.aws_region,

                iCol=self._iCol, tCol=self._tCol,
                _mappers=self._mappers + additionalMappers,

                reprSampleMinNPieces=self._reprSampleMinNPieces,
                reprSampleSize=self._reprSampleSize,

                minNonNullProportion=self._minNonNullProportion,
                outlierTailProportion=self._outlierTailProportion,
                maxNCats=self._maxNCats,
                minProportionByMaxNCats=self._minProportionByMaxNCats,

                **kwargs)

        if inheritCache:
            arrowADF._inheritCache(self)

        if inheritNRows:
            arrowADF._cache.approxNRows = self._cache.approxNRows
            arrowADF._cache.nRows = self._cache.nRows

        return arrowADF

    def reduce(self, *piecePaths, **kwargs):
        # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        _CHUNK_SIZE = 10 ** 5

        nSamplesPerPiece = kwargs.get('nSamplesPerPiece')

        genTAuxCols = kwargs.get('genTAuxCols', True)

        reducer = \
            kwargs.get(
                'reducer',
                lambda results:
                    numpy.vstack(results)
                    if isinstance(results[0], numpy.ndarray)
                    else pandas.concat(
                        objs=results,
                        axis='index',
                        join='outer',
                        ignore_index=False,
                        keys=None,
                        levels=None,
                        names=None,
                        verify_integrity=False,
                        sort=False,
                        copy=False,
                        # FutureWarning:
                        # Sorting because non-concatenation axisis not aligned.
                        # A future version of pandas will change to not sort
                        # by default.
                        # To accept the future behavior, pass 'sort=False'.
                        # To retain the current behavior and
                        # silence the warning, pass 'sort=True'.
                    ))

        verbose = kwargs.pop('verbose', True)

        if not piecePaths:
            piecePaths = self.piecePaths

        results = []

        for piecePath in (tqdm(piecePaths)
                          if verbose and (len(piecePaths) > 1)
                          else piecePaths):
            pieceLocalPath = self.pieceLocalPath(piecePath=piecePath)

            pieceCache = self._PIECE_CACHES[piecePath]

            if pieceCache.nRows is None:
                schema = read_schema(where=pieceLocalPath)

                pieceCache.srcColsExclPartitionKVs = schema.names

                pieceCache.srcColsInclPartitionKVs += schema.names

                self.srcColsInclPartitionKVs.update(schema.names)

                for col in (set(schema.names)
                            .difference(pieceCache.partitionKVs)):
                    pieceCache.srcTypesExclPartitionKVs[col] = \
                        pieceCache.srcTypesInclPartitionKVs[col] = \
                        _arrowType = \
                        schema.field(col).type

                    assert not is_binary(_arrowType), \
                        f'*** {piecePath}: {col} IS OF BINARY TYPE ***'

                    if col in self.srcTypesInclPartitionKVs:
                        assert _arrowType == \
                            self.srcTypesInclPartitionKVs[col], \
                            (f'*** {piecePath} COLUMN {col}: '
                             f'DETECTED TYPE {_arrowType} != '
                             f'{self.srcTypesInclPartitionKVs[col]} ***')

                    else:
                        self.srcTypesInclPartitionKVs[col] = _arrowType

                metadata = read_metadata(where=pieceCache.localPath)
                pieceCache.nCols = metadata.num_columns
                pieceCache.nRows = metadata.num_rows

            cols = kwargs.get('cols')

            cols = (to_iterable(cols, iterable_type=set)
                    if cols
                    else set(pieceCache.srcColsInclPartitionKVs))

            srcCols = cols.intersection(pieceCache.srcColsExclPartitionKVs)

            partitionKeyCols = cols.intersection(pieceCache.partitionKVs)

            if srcCols:
                pieceArrowTable = \
                    read_table(
                        source=pieceLocalPath,
                        # (str, pyarrow.NativeFile, or file-like object) –
                        # If a string passed,
                        # can be a single file name or directory name.
                        # For file-like objects, only read a single file.
                        # Use pyarrow.BufferReader to read
                        # a file contained in a bytes or buffer-like object.

                        columns=list(srcCols),
                        # (list) – If not None,
                        # only these columns will be read from the file.
                        # A column name may be a prefix of a nested field,
                        # e.g. ‘a’ will select ‘a.b’, ‘a.c’, and ‘a.d.e’.
                        # If empty, no columns will be read.
                        # Note that the table will still have the correct
                        # num_rows set despite having no columns.

                        use_threads=True,
                        # (bool, default True) –
                        # Perform multi-threaded column reads.

                        metadata=None,
                        # (FileMetaData) – If separately computed

                        use_pandas_metadata=False,
                        # (bool, default False) –
                        # If True and file has custom pandas schema metadata,
                        # ensure that index columns are also loaded.

                        memory_map=False,
                        # (bool, default False) –
                        # If the source is a file path,
                        # use a memory map to read file,
                        # which can improve performance in some environments.

                        read_dictionary=None,
                        # (list, default None) –
                        # List of names or column paths (for nested types)
                        # to read directly as DictionaryArray.
                        # Only supported for BYTE_ARRAY storage.
                        # To read a flat column as dictionary-encoded
                        # pass the column name.
                        # For nested types, you must pass
                        # the full column “path”, which could be something
                        # like level1.level2.list.item.
                        # Refer to the Parquet file’s schema
                        # to obtain the paths.

                        filesystem=None,
                        # (FileSystem, default None) –
                        # If nothing passed, paths assumed to be found
                        # in the local on-disk filesystem.

                        filters=None,
                        # List[Tuple] or List[List[Tuple]] or None (default))
                        # Rows which do not match the filter predicate
                        # will be removed from scanned data.
                        # Partition keys embedded in a nested directory
                        # structure will be exploited to avoid loading
                        # files at all if they contain no matching rows.
                        # If use_legacy_dataset is True,
                        # filters can only reference partition keys and
                        # only a hive-style directory structure is supported.
                        # When setting use_legacy_dataset to False,
                        # also within-file level filtering and different
                        # partitioning schemes are supported.
                        # Predicates are expressed in disjunctive normal form
                        # (DNF), like [[('x', '=', 0), ...], ...].
                        # DNF allows arbitrary boolean logical combinations
                        # of single column predicates.
                        # The innermost tuples each describe a single column
                        # predicate.
                        # The list of inner predicates is interpreted
                        # as a conjunction (AND),
                        # forming a more selective and multiple column
                        # predicate.
                        # Finally, the most outer list combines these filters
                        # as a disjunction (OR).
                        # Predicates may also be passed as List[Tuple].
                        # This form is interpreted as a single conjunction.
                        # To express OR in predicates, one must use the
                        # (preferred) List[List[Tuple]] notation.

                        buffer_size=0,
                        # (int, default 0) –
                        # If positive, perform read buffering when
                        # deserializing individual column chunks.
                        # Otherwise IO calls are unbuffered.

                        partitioning='hive',
                        # (Partitioning or str or list of str, default "hive")
                        # – The partitioning scheme for a partitioned dataset.
                        # The default of “hive” assumes directory names with
                        # key=value pairs like “/year=2009/month=11”.
                        # In addition, a scheme like “/2009/11” is also
                        # supported, in which case you need to specify the
                        # field names or a full schema.
                        # See the pyarrow.dataset.partitioning() function
                        # for more details.

                        use_legacy_dataset=False,
                        # (bool, default False) –
                        # By default, read_table uses the new Arrow Datasets
                        # API since pyarrow 1.0.0. Among other things,
                        # this allows to pass filters for all columns and
                        # not only the partition keys, enables different
                        # partitioning schemes, etc.
                        # Set to True to use the legacy behaviour.

                        ignore_prefixes=None,
                        # (list, optional) –
                        # Files matching any of these prefixes will be ignored
                        # by the discovery process if use_legacy_dataset=False.
                        # This is matched to the basename of a path.
                        # By default this is [‘.’, ‘_’].
                        # Note that discovery happens only if a directory
                        # is passed as source.

                        pre_buffer=True,
                        # Coalesce and issue file reads in parallel to improve
                        # performance on high-latency filesystems (e.g. S3).
                        # If True, Arrow will use a background I/O thread pool.
                        # This option is only supported for
                        # use_legacy_dataset=False.
                        # If using a filesystem layer that itself performs
                        # readahead (e.g. fsspec’s S3FS),
                        # disable readahead for best results.

                        coerce_int96_timestamp_unit=None,
                        # Cast timestamps that are stored in INT96 format to a
                        # particular resolution (e.g. ‘ms’).
                        # Setting to None is equivalent to ‘ns’ and therefore
                        # INT96 timestamps will be infered as timestamps in
                        # nanoseconds.
                    )

                if nSamplesPerPiece and (nSamplesPerPiece < pieceCache.nRows):
                    intermediateN = (nSamplesPerPiece * pieceCache.nRows) ** .5

                    nChunks = int(math.ceil(pieceCache.nRows / _CHUNK_SIZE))
                    nChunksForIntermediateN = \
                        int(math.ceil(intermediateN / _CHUNK_SIZE))

                    nSamplesPerChunk = \
                        int(math.ceil(
                            nSamplesPerPiece / nChunksForIntermediateN))

                    if nChunksForIntermediateN < nChunks:
                        recordBatches = \
                            pieceArrowTable.to_batches(
                                max_chunksize=_CHUNK_SIZE)

                        nRecordBatches = len(recordBatches)

                        assert nRecordBatches in (nChunks - 1, nChunks), \
                            (f'*** {piecePath}: {nRecordBatches} vs. '
                             f'{nChunks} Record Batches ***')

                        assert nChunksForIntermediateN <= nRecordBatches, \
                            (f'*** {piecePath}: {nChunksForIntermediateN} vs. '
                             f'{nRecordBatches} Record Batches ***')

                        chunkPandasDFs = []

                        for recordBatch in \
                                random_sample(population=recordBatches,
                                              k=nChunksForIntermediateN):
                            chunkPandasDF = \
                                recordBatch.to_pandas(
                                    categories=None,
                                    strings_to_categorical=False,
                                    zero_copy_only=False,
                                    integer_object_nulls=False,
                                    date_as_object=True,
                                    use_threads=True,
                                    deduplicate_objects=False,
                                    ignore_metadata=False)

                            for k in partitionKeyCols:
                                chunkPandasDF[k] = pieceCache.partitionKVs[k]

                            if genTAuxCols and \
                                    (self._tCol in chunkPandasDF.columns):
                                if self._iCol in chunkPandasDF.columns:
                                    try:
                                        chunkPandasDF = \
                                            gen_aux_cols(
                                                df=chunkPandasDF.loc[
                                                    pandas.notnull(
                                                        chunkPandasDF[
                                                            self._iCol]) &   # noqa: E501,W504
                                                    pandas.notnull(
                                                        chunkPandasDF[
                                                            self._tCol])],
                                                i_col=self._iCol,
                                                t_col=self._tCol)

                                    except Exception as err:
                                        print(f'*** {piecePath} ***')

                                        # stackoverflow.com/questions/4825234/
                                        # exception-traceback-is-hidden-if-not-re-raised-immediately
                                        raise err

                                else:
                                    try:
                                        chunkPandasDF = \
                                            gen_aux_cols(
                                                df=chunkPandasDF.loc[
                                                    pandas.notnull(
                                                        chunkPandasDF[
                                                            self._tCol])],
                                                i_col=None,
                                                t_col=self._tCol)

                                    except Exception as err:
                                        print(f'*** {piecePath} ***')

                                        # stackoverflow.com/questions/4825234/
                                        # exception-traceback-is-hidden-if-not-re-raised-immediately
                                        raise err

                            if nSamplesPerChunk < len(chunkPandasDF):
                                chunkPandasDF = \
                                    chunkPandasDF.sample(
                                        n=nSamplesPerChunk,
                                        # Number of items from axis to return.
                                        # Cannot be used with frac.
                                        # Default = 1 if frac = None.
                                        # frac=None,
                                        # Fraction of axis items to return.
                                        # Cannot be used with n.

                                        replace=False,
                                        # Sample with or without replacement.
                                        # Default = False.

                                        weights=None,
                                        # Default None
                                        # results in equal probability
                                        # weighting.
                                        # If passed a Series,
                                        # will align with target object
                                        # on index.
                                        # Index values in weights not found
                                        # in sampled object will be ignored
                                        # and index values in sampled object
                                        # not in weights will be assigned
                                        # weights of zero.
                                        # If called on a DataFrame, will accept
                                        # the name of a column when axis = 0.
                                        # Unless weights are a Series, weights
                                        # must be same length as axis being
                                        # sampled.
                                        # If weights do not sum to 1, they will
                                        # be normalized to sum to 1.
                                        # Missing values in the weights column
                                        # will be treated as zero.
                                        # inf and -inf values not allowed.

                                        random_state=None,
                                        # Seed for the random number generator
                                        # (if int), or numpy RandomState object

                                        axis='index')

                            chunkPandasDFs.append(chunkPandasDF)

                        piecePandasDF = \
                            pandas.concat(
                                objs=chunkPandasDFs,
                                axis='index',
                                join='outer',
                                ignore_index=True,
                                keys=None,
                                levels=None,
                                names=None,
                                verify_integrity=False,
                                copy=False)

                    else:
                        piecePandasDF = \
                            pieceArrowTable.to_pandas(
                                categories=None,
                                strings_to_categorical=False,
                                zero_copy_only=False,
                                integer_object_nulls=False,
                                date_as_object=True,
                                use_threads=True,
                                deduplicate_objects=False,
                                ignore_metadata=False)

                        for k in partitionKeyCols:
                            piecePandasDF[k] = pieceCache.partitionKVs[k]

                        if genTAuxCols and \
                                (self._tCol in piecePandasDF.columns):
                            if self._iCol in piecePandasDF.columns:
                                try:
                                    piecePandasDF = \
                                        gen_aux_cols(
                                            df=piecePandasDF.loc[
                                                pandas.notnull(
                                                    piecePandasDF[
                                                        self._iCol]) &   # noqa: E501,W504
                                                pandas.notnull(
                                                    piecePandasDF[
                                                        self._tCol])],
                                            i_col=self._iCol,
                                            t_col=self._tCol)

                                except Exception as err:
                                    print(f'*** {piecePath} ***')

                                    # stackoverflow.com/questions/4825234/
                                    # exception-traceback-is-hidden-if-not-re-raised-immediately
                                    raise err

                            else:
                                try:
                                    piecePandasDF = \
                                        gen_aux_cols(
                                            df=piecePandasDF.loc[
                                                pandas.notnull(
                                                    piecePandasDF[
                                                        self._tCol])],
                                            i_col=None,
                                            t_col=self._tCol)

                                except Exception as err:
                                    print(f'*** {piecePath} ***')

                                    # stackoverflow.com/questions/4825234/
                                    # exception-traceback-is-hidden-if-not-re-raised-immediately
                                    raise err

                        piecePandasDF = \
                            piecePandasDF.sample(
                                n=nSamplesPerPiece,
                                # Number of items from axis to return.
                                # Cannot be used with frac.
                                # Default = 1 if frac = None.

                                # frac=None,
                                # Fraction of axis items to return.
                                # Cannot be used with n.

                                replace=False,
                                # Sample with or without replacement.
                                # Default = False.

                                weights=None,
                                # Default None
                                # results in equal probability
                                # weighting.
                                # If passed a Series,
                                # will align with target object
                                # on index.
                                # Index values in weights not found
                                # in sampled object will be ignored
                                # and index values in sampled object
                                # not in weights will be assigned
                                # weights of zero.
                                # If called on a DataFrame, will accept
                                # the name of a column when axis = 0.
                                # Unless weights are a Series, weights
                                # must be same length as axis being
                                # sampled.
                                # If weights do not sum to 1, they will
                                # be normalized to sum to 1.
                                # Missing values in the weights column
                                # will be treated as zero.
                                # inf and -inf values not allowed.

                                random_state=None,
                                # Seed for the random number generator
                                # (if int), or numpy RandomState object.

                                axis='index')

                else:
                    piecePandasDF = \
                        pieceArrowTable.to_pandas(
                            categories=None,
                            strings_to_categorical=False,
                            zero_copy_only=False,
                            integer_object_nulls=False,
                            date_as_object=True,
                            use_threads=True,
                            deduplicate_objects=False,
                            ignore_metadata=False)

                    for k in partitionKeyCols:
                        piecePandasDF[k] = pieceCache.partitionKVs[k]

                    if genTAuxCols and (self._tCol in piecePandasDF.columns):
                        if self._iCol in piecePandasDF.columns:
                            try:
                                piecePandasDF = \
                                    gen_aux_cols(
                                        df=piecePandasDF.loc[
                                            pandas.notnull(
                                                piecePandasDF[self._iCol]) &   # noqa: E501,W504
                                            pandas.notnull(
                                                piecePandasDF[self._tCol])],
                                        i_col=self._iCol,
                                        t_col=self._tCol)

                            except Exception as err:
                                print(f'*** {piecePath} ***')

                                # stackoverflow.com/questions/4825234/
                                # exception-traceback-is-hidden-if-not-re-raised-immediately
                                raise err

                        else:
                            try:
                                piecePandasDF = \
                                    gen_aux_cols(
                                        df=piecePandasDF.loc[
                                            pandas.notnull(
                                                piecePandasDF[self._tCol])],
                                        i_col=None,
                                        t_col=self._tCol)

                            except Exception as err:
                                print(f'*** {piecePath} ***')

                                # stackoverflow.com/questions/4825234/
                                # exception-traceback-is-hidden-if-not-re-raised-immediately
                                raise err

            else:
                piecePandasDF = pandas.DataFrame(
                    index=range(nSamplesPerPiece
                                if nSamplesPerPiece and   # noqa: W504
                                (nSamplesPerPiece < pieceCache.nRows)
                                else pieceCache.nRows))

                for k in partitionKeyCols:
                    piecePandasDF[k] = pieceCache.partitionKVs[k]

            for mapper in self._mappers:
                piecePandasDF = mapper(piecePandasDF)

            results.append(piecePandasDF)

        return reducer(results)

    def __getitem__(self, item):
        return self.map(
            mapper=_S3ParquetDataFeeder__getitem__pandasDFTransform(item=item),
            inheritNRows=True)

    def drop(self, *cols, **kwargs):
        return self.map(
            mapper=_S3ParquetDataFeeder__drop__pandasDFTransform(cols=cols),
            inheritNRows=True,
            **kwargs)

    def rename(self, **kwargs):
        """
        Return:
            ``ADF`` with new column names

        Args:
            **kwargs: arguments of the form
            ``newColName`` = ``existingColName``
        """
        renameDict = {}
        remainingKwargs = {}

        for k, v in kwargs.items():
            if v in self.columns:
                if v not in self._T_AUX_COLS:
                    renameDict[v] = k
            else:
                remainingKwargs[k] = v

        return self.map(
            mapper=lambda pandasDF:
                pandasDF.rename(
                    mapper=None,
                    index=None,
                    columns=renameDict,
                    axis='index',
                    copy=False,
                    inplace=False,
                    level=None),
            inheritNRows=True,
            **remainingKwargs)

    def filter(self, *conditions, **kwargs):
        pass   # TODO

    def collect(self, *cols, **kwargs):
        return self.reduce(cols=cols if cols else None, **kwargs)

    def toPandas(self, *cols, **kwargs):
        return self.collect(*cols, **kwargs)

    # =========================
    # KEY (SETTABLE) PROPERTIES
    # -------------------------
    # iCol
    # tCol

    @property
    def iCol(self):
        return self._iCol

    @iCol.setter
    def iCol(self, iCol):
        if iCol != self._iCol:
            self._iCol = iCol

            if iCol is None:
                self.hasTS = False
            else:
                assert iCol
                self.hasTS = bool(self._tCol)

    @iCol.deleter
    def iCol(self):
        self._iCol = None
        self.hasTS = False

    @property
    def tCol(self):
        return self._tCol

    @tCol.setter
    def tCol(self, tCol):
        if tCol != self._tCol:
            self._tCol = tCol

            if tCol is None:
                self.hasTS = False
            else:
                assert tCol
                self.hasTS = bool(self._iCol)

    @tCol.deleter
    def tCol(self):
        self._tCol = None
        self.hasTS = False

    # ===========
    # REPR SAMPLE
    # -----------
    # prelimReprSamplePiecePaths
    # reprSamplePiecePaths
    # _assignReprSample

    @property
    def prelimReprSamplePiecePaths(self):
        if self._cache.prelimReprSamplePiecePaths is None:
            self._cache.prelimReprSamplePiecePaths = \
                random_sample(population=self.piecePaths,
                              k=self._reprSampleMinNPieces)

        return self._cache.prelimReprSamplePiecePaths

    @property
    def reprSamplePiecePaths(self):
        if self._cache.reprSamplePiecePaths is None:
            reprSampleNPieces = \
                int(math.ceil(
                    ((min(self._reprSampleSize, self.approxNRows) /   # noqa: E501,W504
                     self.approxNRows) ** .5) * self.nPieces))

            self._cache.reprSamplePiecePaths = \
                self._cache.prelimReprSamplePiecePaths + \
                (random_sample(
                    population=self.piecePaths,
                    k=reprSampleNPieces - self._reprSampleMinNPieces)
                 if reprSampleNPieces > self._reprSampleMinNPieces
                 else [])

        return self._cache.reprSamplePiecePaths

    def _assignReprSample(self):
        self._cache.reprSample = \
            self.sample(
                n=self._reprSampleSize,
                piecePaths=self.reprSamplePiecePaths,
                verbose=True)

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

    def _read_metadata_and_schema(self, piecePath):
        pieceLocalPath = self.pieceLocalPath(piecePath=piecePath)

        pieceCache = self._PIECE_CACHES[piecePath]

        if pieceCache.nRows is None:
            schema = read_schema(where=pieceLocalPath)

            pieceCache.srcColsExclPartitionKVs = schema.names

            pieceCache.srcColsInclPartitionKVs += schema.names

            self.srcColsInclPartitionKVs.update(schema.names)

            for col in set(schema.names).difference(pieceCache.partitionKVs):
                pieceCache.srcTypesExclPartitionKVs[col] = \
                    pieceCache.srcTypesInclPartitionKVs[col] = \
                    _arrowType = \
                    schema.field(col).type

                assert not is_binary(_arrowType), \
                    f'*** {piecePath}: {col} IS OF BINARY TYPE ***'

                if col in self.srcTypesInclPartitionKVs:
                    assert _arrowType == self.srcTypesInclPartitionKVs[col], \
                        (f'*** {piecePath} COLUMN {col}: '
                         f'DETECTED TYPE {_arrowType} != '
                         f'{self.srcTypesInclPartitionKVs[col]} ***')

                else:
                    self.srcTypesInclPartitionKVs[col] = _arrowType

            metadata = read_metadata(where=pieceCache.localPath)
            pieceCache.nCols = metadata.num_columns
            pieceCache.nRows = metadata.num_rows

        return pieceCache

    @property
    def approxNRows(self):
        if self._cache.approxNRows is None:
            self.stdout_logger.info('Counting Approx. No. of Rows...')

            self._cache.approxNRows = \
                self.nPieces \
                * sum(self._read_metadata_and_schema(piecePath=piecePath).nRows
                      for piecePath in
                      (tqdm(self.prelimReprSamplePiecePaths)
                       if len(self.prelimReprSamplePiecePaths) > 1
                       else self.prelimReprSamplePiecePaths)) \
                / self._reprSampleMinNPieces

        return self._cache.approxNRows

    @property
    def nRows(self):
        if self._cache.nRows is None:
            self.stdout_logger.info('Counting No. of Rows...')

            self._cache.nRows = \
                sum(self._read_metadata_and_schema(piecePath=piecePath).nRows
                    for piecePath in (tqdm(self.piecePaths)
                                      if self.nPieces > 1
                                      else self.piecePaths))

        return self._cache.nRows

    def __len__(self):
        return (self._cache.nRows
                if self._cache.nRows
                else self.approxNRows)

    @property
    def columns(self):
        return list(self.srcColsInclPartitionKVs) + \
            (list(self._T_AUX_COLS
                  if self._iCol
                  else self._T_COMPONENT_AUX_COLS)
             if self._tCol
             else [])

    @property
    def types(self):
        if self._tCol:
            _types = Namespace(
                **{col: self._T_AUX_COL_ARROW_TYPES[col]
                   for col in (self._T_AUX_COLS
                               if self._iCol
                               else self._T_COMPONENT_AUX_COLS)})

            for col, arrowType in self.srcTypesInclPartitionKVs.items():
                _types[col] = arrowType

            return _types

        return self.srcTypesInclPartitionKVs

    def type(self, col):
        return self.types[col]

    def typeIsNum(self, col):
        return is_num(self.type(col))

    def typeIsComplex(self, col):
        return is_complex(self.type(col))

    # =============
    # COLUMN GROUPS
    # -------------
    # indexCols
    # tRelAuxCols
    # possibleFeatureContentCols
    # possibleCatContentCols

    @property
    def indexCols(self):
        return (((self._iCol,) if self._iCol else ()) +   # noqa: W504
                ((self._dCol,) if self._dCol else ()) +   # noqa: W504
                ((self._tCol,) if self._tCol else ()))

    @property
    def tRelAuxCols(self):
        return ((self._T_ORD_COL, self._T_DELTA_COL)
                if self.hasTS
                else ())

    @property
    def possibleFeatureContentCols(self):
        def chk(t):
            return is_boolean(t) or is_string(t) or is_num(t)

        return tuple(col
                     for col in self.contentCols
                     if chk(self.type(col)))

    @property
    def possibleCatContentCols(self):
        return tuple(col
                     for col in self.contentCols
                     if is_possible_cat(self.type(col)))

    # **************
    # SUBSET METHODS
    # _subset
    # filterByPartitionKeys
    # sample
    # gen

    def _subset(self, *piecePaths, **kwargs):
        if piecePaths:
            assert self.piecePaths.issuperset(piecePaths)

            nPiecePaths = len(piecePaths)

            if nPiecePaths == self.nPieces:
                return self

            if nPiecePaths > 1:
                verbose = kwargs.pop('verbose', True)

                subsetDirS3Key = os.path.join(self.tmpDirS3Key,
                                              str(uuid.uuid4()))

                _pathPlusSepLen = len(self.path) + 1

                for piecePath in (tqdm(piecePaths) if verbose else piecePaths):
                    pieceSubPath = piecePath[_pathPlusSepLen:]

                    _from_key = os.path.join(self.pathS3Key, pieceSubPath)
                    _to_key = os.path.join(subsetDirS3Key, pieceSubPath)

                    try:
                        self.s3Client.copy(
                            CopySource=dict(Bucket=self.s3Bucket,
                                            Key=_from_key),
                            Bucket=self.s3Bucket,
                            Key=_to_key)

                    except Exception as err:
                        print(f'*** FAILED TO COPY FROM "{_from_key}" '
                              f'TO "{_to_key}" ***')

                        raise err

                subsetPath = os.path.join(f's3://{self.s3Bucket}',
                                          subsetDirS3Key)

            else:
                subsetPath = piecePaths[0]

            return S3ParquetDataFeeder(
                path=subsetPath,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_region=self.aws_region,

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

    def filterByPartitionKeys(self, *filterCriteriaTuples, **kwargs):
        filterCriteria = {}

        _samplePiecePath = next(iter(self.piecePaths))

        for filterCriteriaTuple in filterCriteriaTuples:
            assert isinstance(filterCriteriaTuple, (list, tuple))
            filterCriteriaTupleLen = len(filterCriteriaTuple)

            col = filterCriteriaTuple[0]

            if f'{col}=' in _samplePiecePath:
                if filterCriteriaTupleLen == 2:
                    fromVal = toVal = None
                    inSet = {str(v)
                             for v in to_iterable(filterCriteriaTuple[1])}

                elif filterCriteriaTupleLen == 3:
                    fromVal = filterCriteriaTuple[1]
                    if fromVal is not None:
                        fromVal = str(fromVal)

                    toVal = filterCriteriaTuple[2]
                    if toVal is not None:
                        toVal = str(toVal)

                    inSet = None

                else:
                    raise ValueError(
                        f'*** {type(self)} FILTER CRITERIA MUST BE EITHER '
                        '(<colName>, <fromVal>, <toVal>) OR '
                        '(<colName>, <inValsSet>) ***')

                filterCriteria[col] = fromVal, toVal, inSet

        if filterCriteria:
            piecePaths = set()

            for piecePath in self.piecePaths:
                chk = True

                for col, (fromVal, toVal, inSet) in filterCriteria.items():
                    v = re.search(f'{col}=(.*?)/', piecePath).group(1)

                    if ((fromVal is not None) and (v < fromVal)) or \
                            ((toVal is not None) and (v > toVal)) or \
                            ((inSet is not None) and (v not in inSet)):
                        chk = False
                        break

                if chk:
                    piecePaths.add(piecePath)

            assert piecePaths, \
                (f'*** {self}: NO PIECE PATHS SATISFYING '
                 f'FILTER CRITERIA {filterCriteria} ***')

            if h1st_util.debug.ON:
                self.stdout_logger.debug(
                    msg=(f'*** {len(piecePaths)} PIECES SATISFYING '
                         f'FILTERING CRITERIA: {filterCriteria} ***'))

            return self._subset(*piecePaths, **kwargs)

        return self

    def sample(self, *cols, **kwargs):
        n = kwargs.pop('n', self._DEFAULT_REPR_SAMPLE_SIZE)

        piecePaths = kwargs.pop('piecePaths', None)

        verbose = kwargs.pop('verbose', True)

        if piecePaths:
            nSamplePieces = len(piecePaths)

        else:
            minNPieces = kwargs.pop('minNPieces', self._reprSampleMinNPieces)
            maxNPieces = kwargs.pop('maxNPieces', None)

            nSamplePieces = \
                max(int(math.ceil(
                    ((min(n, self.approxNRows) / self.approxNRows) ** .5)
                    * self.nPieces)),   # noqa: W503
                    minNPieces) \
                if (self.nPieces > 1) and \
                ((maxNPieces is None) or (maxNPieces > 1)) \
                else 1

            if maxNPieces:
                nSamplePieces = min(nSamplePieces, maxNPieces)

            if nSamplePieces < self.nPieces:
                piecePaths = random_sample(population=self.piecePaths,
                                           k=nSamplePieces)

            else:
                nSamplePieces = self.nPieces
                piecePaths = self.piecePaths

        if verbose or h1st_util.debug.ON:
            self.stdout_logger.info(
                f"Sampling {n:,} Rows{f' of Columns {cols}' if cols else ''} "
                f'from {nSamplePieces:,} Pieces...')

        return self.reduce(
            *piecePaths,
            cols=cols,
            nSamplesPerPiece=int(math.ceil(n / nSamplePieces)),
            verbose=verbose,
            **kwargs)

    # ****************
    # COLUMN PROFILING
    # count
    # nonNullProportion
    # distinct
    # quantile
    # sampleStat
    # outlierRstStat / outlierRstMin / outlierRstMax
    # profile

    def count(self, *cols, **kwargs):
        """
        Return:
            - If 1 column name is given,
            return its corresponding non-``NULL`` count

            - If multiple column names are given,
            return a {``col``: corresponding non-``NULL`` count} *dict*

            - If no column names are given,
            return a {``col``: corresponding non-``NULL`` count} *dict*
            for all columns

        Args:
             *cols (str): column name(s)

             **kwargs:
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
                           if h1st_util.debug.ON
                           else kwargs.get('verbose'))

                if verbose:
                    tic = time.time()

                self._cache.count[col] = result = \
                    self[col] \
                    .map(
                        mapper=   # noqa: E251
                        ((lambda series:
                            series.notnull().sum(skipna=True, min_count=0))
                            if pandas.isnull(upperNumericNull)
                            else (lambda series:
                                  (series < upperNumericNull)
                                  .sum(skipna=True, min_count=0)))
                        if pandas.isnull(lowerNumericNull)
                        else ((lambda series:
                               (series > lowerNumericNull)
                               .sum(skipna=True, min_count=0))
                              if pandas.isnull(upperNumericNull)
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
                    self.stdout_logger.info(
                        msg=(f'No. of Non-NULLs of Column "{col}" = '
                             f'{result:,}   <{toc - tic:,.1f} s>'))

            return self._cache.count[col]

        return (
            (pandasDF[col].notnull().sum(skipna=True, min_count=0)
             if pandas.isnull(upperNumericNull)
             else (pandasDF[col] < upperNumericNull).sum(skipna=True,
                                                         min_count=0))

            if pandas.isnull(lowerNumericNull)

            else ((pandasDF[col] > lowerNumericNull).sum(skipna=True,
                                                         min_count=0)

                  if pandas.isnull(upperNumericNull)

                  else pandasDF[col].between(left=lowerNumericNull,
                                             right=upperNumericNull,
                                             inclusive='neither').sum(
                                                 skipna=True,
                                                 min_count=0))
        )

    def nonNullProportion(self, *cols, **kwargs):
        """
        Return:
            - If 1 column name is given,
            return its *approximate* non-``NULL`` proportion

            - If multiple column names are given,
            return {``col``: approximate non-``NULL`` proportion} *dict*

            - If no column names are given,
            return {``col``: approximate non-``NULL`` proportion}
            *dict* for all columns

        Args:
             *cols (str): column name(s)

             **kwargs:
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
        """
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

    @lru_cache()
    def quantile(self, *cols, **kwargs):
        if len(cols) > 1:
            return Namespace(**{col: self.quantile(col, **kwargs)
                                for col in cols})

        col = cols[0]

        return self[col] \
            .reduce(cols=col) \
            .quantile(
                q=kwargs.get('q', .5),
                interpolation='linear')

    def sampleStat(self, *cols, **kwargs):
        """
        *Approximate* measurements of a certain statistic
        on **numerical** columns

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
            cache = getattr(self._cache, s)

            if col not in cache:
                verbose = True \
                    if h1st_util.debug.ON \
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
                    self.stdout_logger.info(
                        msg=(f'Sample {capitalizedStatName} for '
                             f'Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                cache[col] = result

            return cache[col]

        raise ValueError(
            f'{self}.sampleStat({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def outlierRstStat(self, *cols, **kwargs):
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
            cache = getattr(self._cache, s)

            if col not in cache:
                verbose = True \
                    if h1st_util.debug.ON \
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

                if pandas.isnull(result):
                    self.stdout_logger.warning(
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
                    self.stdout_logger.info(
                        msg=(f'Outlier-Resistant {capitalizedStatName}'
                             f' for Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                cache[col] = result

            return cache[col]

        raise ValueError(
            f'{self}.outlierRstStat({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def outlierRstMin(self, *cols, **kwargs):
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
                    if h1st_util.debug.ON \
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
                    self.stdout_logger.info(
                        msg=(f'Outlier-Resistant Min of Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                self._cache.outlierRstMin[col] = result

            return self._cache.outlierRstMin[col]

        raise ValueError(
            f'{self}.outlierRstMin({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def outlierRstMax(self, *cols, **kwargs):
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
                           if h1st_util.debug.ON
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
                    self.stdout_logger.info(
                        msg=(f'Outlier-Resistant Max of Column "{col}" = '
                             f'{result:,.3g}   <{toc - tic:,.1f} s>'))

                self._cache.outlierRstMax[col] = result

            return self._cache.outlierRstMax[col]

        raise ValueError(
            f'{self}.outlierRstMax({col}, ...): '
            f'Column "{col}" Is Not of Numeric Type')

    def profile(self, *cols, **kwargs):
        """
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
                   if h1st_util.debug.ON
                   else kwargs.get('verbose'))

        if verbose:
            msg = f'Profiling Column "{col}"...'
            self.stdout_logger.info(msg)
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
                    pandas.Series(
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
                        numpy.isnan(quantilesOfInterest)] = \
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
            self.stdout_logger.info(
                msg + f' done!   <{toc - tic:,.1f} s>')

        return (Namespace(**{col: profile})
                if asDict
                else profile)

    # *********
    # DATA PREP
    # fillna
    # prep

    def fillna(self, *cols, **kwargs):
        """
        Fill/interpolate ``NULL``/``NaN`` values

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
        if h1st_util.debug.ON:
            verbose = True

        if loadPath:
            if verbose:
                message = ('Loading NULL-Filling SQL Statement '
                           f'from Path "{loadPath}"...')
                self.stdout_logger.info(message)
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

            cols.difference_update(self.indexCols + (self._T_ORD_COL,))

            nulls = kwargs.pop('nulls', {})

            for col in cols:
                if col in nulls:
                    colNulls = nulls[col]

                    assert (isinstance(colNulls, (list, tuple)) and   # noqa: E501,W504
                            (len(colNulls) == 2) and   # noqa: W504
                            ((colNulls[0] is None) or   # noqa: W504
                             isinstance(colNulls[0], PY_NUM_TYPES)) and   # noqa: E501,W504
                            ((colNulls[1] is None) or   # noqa: W504
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
                self.stdout_logger.info(message)
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
                                 f"{', '.join(s.upper() for s in _TS_FILL_METHODS)} "   # noqa: E501
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
                                      or (window == 'partition')   # noqa: W503
                                      else 'mean'),
                                outlierTails=colOutlierTails,
                                verbose=verbose > 1)

                    elif isinstance(value, dict):
                        colFallBackVal = value.get(col)
                        if not isinstance(colFallBackVal, _NUM_CLASSES):
                            colFallBackVal = None

                    elif isinstance(value, _NUM_CLASSES):
                        colFallBackVal = value

                else:
                    isNum = False

                    if isinstance(value, dict):
                        colFallBackVal = value.get(col)
                        if isinstance(colFallBackVal, _NUM_CLASSES):
                            colFallBackVal = None

                    elif not isinstance(value, _NUM_CLASSES):
                        colFallBackVal = value

                if pandas.notnull(colFallBackVal):
                    fallbackStrs = \
                        [f"'{colFallBackVal}'"
                         if is_string(colType) and   # noqa: W504
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

                        dict(
                            SQL="COALESCE(CASE WHEN (STRING({0}) = 'NaN'){1}{2}{3}{4} THEN NULL ELSE {0} END, {5})"   # noqa: E501
                                .format(
                                    col,
                                    '' if lowerNull is None
                                    else f' OR ({col} <= {lowerNull})',
                                    '' if upperNull is None
                                    else f' OR ({col} >= {upperNull})',
                                    f' OR ({col} < {self.outlierRstMin(col)})'
                                    if isNum and (col in fillOutliers)
                                    and fixLowerTail   # noqa: W503
                                    else '',
                                    f' OR ({col} > {self.outlierRstMax(col)})'
                                    if isNum and (col in fillOutliers)
                                    and fixUpperTail   # noqa: W503
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
                self.stdout_logger.info(msg)
                _tic = time.time()

            fs.mkdir(
                dir=savePath,
                hdfs=False)

            with open(os.path.join(savePath,
                                   self._NULL_FILL_SQL_STATEMENT_FILE_NAME),
                      mode='w',
                      encoding='utf-8') as f:
                json.dump(sqlStatement, f, indent=2)

            if verbose:
                _toc = time.time()
                self.stdout_logger.info(
                    msg + f' done!   <{_toc - _tic:,.1f} s>')

        arrowADF = \
            self.map(
                mapper=_S3ParquetDataFeeder__fillna__pandasDFTransform(
                    nullFillDetails=details),
                inheritNRows=True,
                **kwargs)

        arrowADF._inheritCache(
            self,
            *(() if loadPath else cols))

        arrowADF._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdout_logger.info(
                message + f' done!   <{((toc - tic) / 60):,.1f} m>')

        return (((arrowADF, details, sqlStatement)
                 if returnSQLStatement
                 else (arrowADF, details))
                if returnDetails
                else arrowADF)

    def prep(self, *cols, **kwargs):
        """
        Pre-process ``ADF``'s selected column(s) in standard ways:
            - One-hot-encode categorical columns
            - Scale numerical columns

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
        if h1st_util.debug.ON:
            verbose = True

        if loadPath:
            if verbose:
                message = ('Loading & Applying Data Transformations '
                           f'from Path "{loadPath}"...')
                self.stdout_logger.info(message)
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

                    hdfsDirExists = \
                        h1st_util.data_backend.hdfs.test(
                            path=loadPath,
                            exists=True,
                            directory=True)

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
                    if self.suffNonNull(col) and   # noqa: W504
                    (len(profile[col].distinctProportions.loc[
                        # (profile[col].distinctProportions.index != '') &
                        # FutureWarning:
                        # elementwise comparison failed;
                        # returning scalar instead,
                        # but in the future will perform
                        # elementwise comparison
                        pandas.notnull(
                            profile[col].distinctProportions.index)]) > 1
                     )}

            if not cols:
                return self.copy()

            catCols = [
                col
                for col in (cols
                            .intersection(self.possibleCatCols)
                            .difference(forceNum))
                if (col in forceCat) or   # noqa: W504
                (profile[col].distinctProportions
                 .iloc[:self._maxNCats[col]].sum()
                 >= self._minProportionByMaxNCats[col])]   # noqa: W503

            numCols = [col
                       for col in cols.difference(catCols)
                       if self.typeIsNum(col)]

            cols = catCols + numCols

            if verbose:
                message = \
                    'Prepping Columns {}...'.format(
                        ', '.join(f'"{col}"' for col in cols))
                self.stdout_logger.info(message)
                tic = time.time()

            prepSqlItems = {}

            catOrigToPrepColMap = \
                dict(__OHE__=False,
                     __SCALE__=scaleCat)

            if catCols:
                if verbose:
                    msg = 'Transforming Categorical Features {}...'.format(
                        ', '.join(f'"{catCol}"' for catCol in catCols))
                    self.stdout_logger.info(msg)
                    _tic = time.time()

                catIdxCols = []

                if scaleCat:
                    catScaledIdxCols = []

                for catCol in catCols:
                    catIdxCol = (self._CAT_IDX_PREFIX +   # noqa: W504
                                 catCol +   # noqa: W504
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
                            if pandas.notnull(cat) and   # noqa: W504
                            ((cat != '') if isStr else numpy.isfinite(cat))]

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
                        catPrepCol = (self._MIN_MAX_SCL_PREFIX +   # noqa: W504
                                      self._CAT_IDX_PREFIX +   # noqa: W504
                                      catCol +   # noqa: W504
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
                    self.stdout_logger.info(
                        msg + f' done!   <{_toc - tic:,.1f} s>')

            numOrigToPrepColMap = \
                dict(__SCALER__=scaler)

            if numCols:
                numScaledCols = []

                if verbose:
                    msg = 'Transforming Numerical Features {}...'.format(
                        ', '.join(f'"{numCol}"' for numCol in numCols))
                    self.stdout_logger.info(msg)
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
                        assert numpy.allclose(numColNullFillValue,
                                              self.outlierRstStat(numCol))

                        if scaler:
                            if scaler == 'standard':
                                scaledCol = (
                                    self._STD_SCL_PREFIX +   # noqa: W504
                                    numCol +   # noqa: W504
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
                                scaledCol = (self._MAX_ABS_SCL_PREFIX +   # noqa: E501,W504
                                             numCol +   # noqa: W504
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
                                scaledCol = (self._MIN_MAX_SCL_PREFIX +   # noqa: E501,W504
                                             numCol +   # noqa: W504
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
                            scaledCol = (self._NULL_FILL_PREFIX +  # noqa: W504
                                         numCol +   # noqa: W504
                                         self._PREP_SUFFIX)

                            prepSqlItems[scaledCol] = numColSqlItem

                            numOrigToPrepColMap[numCol] = \
                                [scaledCol,

                                 dict(Nulls=numColNulls,
                                      NullFillValue=numColNullFillValue)]

                        numScaledCols.append(scaledCol)

                if verbose:
                    _toc = time.time()
                    self.stdout_logger.info(
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
                self.stdout_logger.info(msg)
                _tic = time.time()

            fs.mkdir(
                dir=savePath,
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
                self.stdout_logger.info(
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
                       if (catCol not in ('__OHE__', '__SCALE__')) and   # noqa: E501,W504
                       isinstance(catPrepColDetails, list) and   # noqa: W504
                       (len(catPrepColDetails) == 2)) + \
                sorted(numPrepColDetails[0]
                       for numCol, numPrepColDetails in
                       numOrigToPrepColMap.items()
                       if (numCol not in ('__TS_WINDOW_CLAUSE__',
                                          '__SCALER__')) and   # noqa: W504
                       isinstance(numPrepColDetails, list) and   # noqa: W504
                       (len(numPrepColDetails) == 2))

        else:
            colsToKeep = \
                self.columns + \
                (([catPrepColDetails[0]
                   for catCol, catPrepColDetails in catOrigToPrepColMap.items()
                   if (catCol not in ('__OHE__', '__SCALE__')) and   # noqa: E501,W504
                   isinstance(catPrepColDetails, list) and   # noqa: W504
                   (len(catPrepColDetails) == 2)] +   # noqa: W504
                  [numPrepColDetails[0]
                   for numCol, numPrepColDetails in numOrigToPrepColMap.items()
                   if (numCol not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and   # noqa: E501,W504
                   isinstance(numPrepColDetails, list) and   # noqa: W504
                   (len(numPrepColDetails) == 2)])
                 if loadPath
                 else (((catScaledIdxCols
                         if scaleCat
                         else catIdxCols)
                        if catCols
                        else []) +   # noqa: W504
                       (numScaledCols
                        if numCols
                        else [])))

        missingCatCols = \
            set(catOrigToPrepColMap) \
            .difference(
                self.columns +   # noqa: W504
                ['__OHE__', '__SCALE__'])

        missingNumCols = \
            set(numOrigToPrepColMap) \
            .difference(
                self.columns +   # noqa: W504
                ['__TS_WINDOW_CLAUSE__', '__SCALER__'])

        missingCols = missingCatCols | missingNumCols

        addCols = {}

        if missingCols:
            if h1st_util.debug.ON:
                self.stdout_logger.debug(
                    msg=f'*** FILLING MISSING COLS {missingCols} ***')

            for missingCol in missingCols:
                addCols[missingCol] = \
                    numpy.nan \
                    if missingCol in missingCatCols \
                    else numOrigToPrepColMap[missingCol][1]['NullFillValue']

                if not returnNumPy:
                    colsToKeep.append(missingCol)

        arrowADF = \
            self.map(
                mapper=_S3ParquetDataFeeder__prep__pandasDFTransform(
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
            self.stdout_logger.info(
                message + f' done!   <{((toc - tic) / 60):,.1f} m>')

        return (((arrowADF, catOrigToPrepColMap, numOrigToPrepColMap,
                  sqlStatement)
                 if returnSQLStatement
                 else (arrowADF, catOrigToPrepColMap, numOrigToPrepColMap))
                if returnOrigToPrepColMaps
                else arrowADF)

    # *******************************
    # ITERATIVE GENERATION / SAMPLING
    # gen

    def gen(self, *args, **kwargs):
        piecePaths = kwargs.get('piecePaths', self.piecePaths)

        return _S3ParquetDataFeeder__gen(
            args=args,
            piecePaths=piecePaths,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            partitionKVs={piecePath: self._PIECE_CACHES[piecePath].partitionKVs
                          for piecePath in piecePaths},
            iCol=self._iCol, tCol=self._tCol,
            possibleFeatureTAuxCols=self.possibleFeatureTAuxCols,
            contentCols=self.contentCols,
            pandasDFTransforms=self._mappers,
            filterConditions=kwargs.get('filter', {}),
            n=kwargs.get('n', 512),
            sampleN=kwargs.get('sampleN', 10 ** (4 if self.hasTS else 5)),
            pad=kwargs.get('pad', numpy.nan),
            anon=kwargs.get('anon', True),
            nThreads=kwargs.get('nThreads', 1))

    # ****
    # MISC
    # split
    # copyToPath

    def split(self, *weights, **kwargs):
        if (not weights) or weights == (1,):
            return self

        nWeights = len(weights)
        cumuWeights = numpy.cumsum(weights) / sum(weights)

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

    def copyToPath(self, path, verbose=True):
        assert path.startswith('s3://')

        s3.sync(
            from_dir_path=self.path,
            to_dir_path=path,
            access_key_id=self.aws_access_key_id,
            secret_access_key=self.aws_secret_access_key,
            delete=True, quiet=True,
            verbose=verbose)

    def schemaDiff(self, parquet_data_feeder):
        return {col: (self.srcTypesInclPartitionKVs[col],
                      parquet_data_feeder.srcTypesInclPartitionKVs[col])
                for col in (set(self.srcTypesInclPartitionKVs)
                            .intersection(
                                parquet_data_feeder.srcTypesInclPartitionKVs))
                if self.srcTypesInclPartitionKVs[col] !=   # noqa: W504
                parquet_data_feeder.srcTypesInclPartitionKVs[col]}
