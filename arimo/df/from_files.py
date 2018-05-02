from __future__ import division, print_function

import abc
from argparse import Namespace as _Namespace
from collections import Counter
import datetime
import json
import math
import multiprocessing
import numpy
import os
import pandas
import psutil
import random
import re
import time
import tqdm

import six

if six.PY2:
    from functools32 import lru_cache
    from urlparse import urlparse
    _NUM_CLASSES = int, long, float
    _STR_CLASSES = str, unicode

else:
    from functools import lru_cache
    from urllib.parse import urlparse
    _NUM_CLASSES = int, float
    _STR_CLASSES = str

from pyarrow.filesystem import LocalFileSystem
from pyarrow.hdfs import HadoopFileSystem
from pyarrow.parquet import ParquetDataset, read_metadata, read_schema, read_table
from s3fs import S3FileSystem

from arimo.df import _DFABC
from arimo.util import DefaultDict, fs, Namespace
from arimo.util.aws import s3
from arimo.util.date_time import gen_aux_cols, DATE_COL
from arimo.util.decor import enable_inplace, _docstr_settable_property, _docstr_verbose
from arimo.util.types.arrow import is_boolean, is_complex, is_num, is_possible_cat, is_string
import arimo.debug


class _ArrowDFABC(_DFABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_REPR_SAMPLE_N_PIECES = 100

    # file systems
    _LOCAL_ARROW_FS = LocalFileSystem()

    _HDFS_ARROW_FS = \
        HadoopFileSystem(
            host='default',
            port=0,
            user=None,
            kerb_ticket=None,
            driver='libhdfs') \
        if fs._ON_LINUX_CLUSTER_WITH_HDFS \
        else None

    _S3_FSs = {}

    @classmethod
    def _s3FS(cls, key=None, secret=None):
        keyPair = key, secret
        if keyPair not in cls._S3_FSs:
            cls._S3_FSs[keyPair] = \
                S3FileSystem(
                    key=key,
                    secret=secret)
        return cls._S3_FSs[keyPair]

    @property
    def reprSampleNPieces(self):
        return self._reprSampleNPieces

    @reprSampleNPieces.setter
    def reprSampleNPieces(self, reprSampleNPieces):
        if (reprSampleNPieces <= self.nPieces) and (reprSampleNPieces != self._reprSampleNPieces):
            self._reprSampleNPieces = reprSampleNPieces

    @reprSampleNPieces.deleter
    def reprSampleNPieces(self):
        self._reprSampleNPieces = min(self._DEFAULT_REPR_SAMPLE_N_PIECES, self.nPieces)


@enable_inplace
class ArrowDF(_ArrowDFABC):
    # "inplace-able" methods
    _INPLACE_ABLE = \
        'rename', \
        'sample', \
        '_subset', \
        'drop', \
        'fillna', \
        'filter', \
        'filterByPartitionKeys', \
        'prep', \
        'transform'

    _CACHE = {}
    
    _PIECE_CACHES = {}

    def __init__(
            self, path=None, reCache=False,
            aws_access_key_id=None, aws_secret_access_key=None,

            defaultMapper=None,

            iCol=None, tCol=None,

            reprSampleNPieces=_ArrowDFABC._DEFAULT_REPR_SAMPLE_N_PIECES,
            reprSampleSize=_ArrowDFABC._DEFAULT_REPR_SAMPLE_SIZE,

            minNonNullProportion=DefaultDict(_ArrowDFABC._DEFAULT_MIN_NON_NULL_PROPORTION),
            outlierTailProportion=DefaultDict(_ArrowDFABC._DEFAULT_OUTLIER_TAIL_PROPORTION),
            maxNCats=DefaultDict(_ArrowDFABC._DEFAULT_MAX_N_CATS),
            minProportionByMaxNCats=DefaultDict(_ArrowDFABC._DEFAULT_MIN_PROPORTION_BY_MAX_N_CATS),

            verbose=True):
        if verbose or arimo.debug.ON:
            logger = self.class_stdout_logger()

        self.path = path

        if (not reCache) and (path in self._CACHE):
            _cache = self._CACHE[path]

        else:
            self._CACHE[path] = _cache = Namespace()

        if _cache:
            if arimo.debug.ON:
                logger.debug('*** RETRIEVING CACHE FOR "{}" ***'.format(path))

        else:
            if path.startswith('s3'):
                self.s3Client = \
                    _cache.s3Client = \
                    s3.client(
                        access_key_id=aws_access_key_id,
                        secret_access_key=aws_secret_access_key)

                _parsedURL = urlparse(url=path, scheme='', allow_fragments=True)
                _cache.s3Bucket = _parsedURL.netloc
                _cache.pathS3Key = _parsedURL.path[1:]

                _cache.tmpDirS3Key = self._TMP_DIR_PATH.strip('/')

                _cache.tmpDirPath = \
                    os.path.join(
                        's3://{}'.format(_cache.s3Bucket),
                        _cache.tmpDirS3Key)

            else:
                _cache.s3Client = _cache.s3Bucket = _cache.tmpDirS3Key = None
                _cache.tmpDirPath = self._TMP_DIR_PATH

            if verbose:
                msg = 'Loading Arrow Dataset from "{}"...'.format(path)
                logger.info(msg)
                tic = time.time()

            _cache._arrowDS = \
                ParquetDataset(
                    path_or_paths=path,
                    filesystem=
                        self._s3FS(
                            key=aws_access_key_id,
                            secret=aws_secret_access_key)
                        if path.startswith('s3')
                        else (self._HDFS_ARROW_FS
                              if path.startswith('hdfs:')
                              else self._LOCAL_ARROW_FS),
                    schema=None, validate_schema=False, metadata=None,
                    split_row_groups=False)

            if verbose:
                toc = time.time()
                logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

            _cache.nPieces = len(_cache._arrowDS.pieces)

            if _cache.nPieces:
                _cache.piecePaths = \
                    {piece.path
                     for piece in _cache._arrowDS.pieces}

            else:
                _cache.nPieces = 1
                _cache.piecePaths = {path}

            for i, piecePath in enumerate(_cache.piecePaths):
                if i and (piecePath in self._PIECE_CACHES):
                    pieceCache = self._PIECE_CACHES[piecePath]

                    pieceCache.arrowDFs.add(self)

                    _cache.columns.update(pieceCache.columns)

                    for col, arrowType in pieceCache.types.items():
                        if col in _cache.types:
                            assert arrowType == self.types[col], \
                                '*** {} COLUMN {}: DETECTED TYPE {} != {} ***'.format(
                                    piecePath, col, arrowType, self.types[col])

                        else:
                            _cache.types[col] = arrowType

                elif i:
                    self._PIECE_CACHES[piecePath] = \
                        Namespace(
                            arrowDFs={self},
                            localOrHDFSPath=None
                                if path.startswith('s3')
                                else piecePath,
                            columns=(),
                            types=Namespace(),
                            nRows=None)

                else:
                    pieceLocalOrHDFSPath = self.pieceLocalOrHDFSPath(piecePath=piecePath)

                    schema = read_schema(where=pieceLocalOrHDFSPath)

                    _cache.columns = set(schema.names)

                    _cache.types = \
                        Namespace(
                            **{col: schema.field_by_name(col).type
                               for col in schema.names})

                    self._PIECE_CACHES[piecePath] = \
                        Namespace(
                            arrowDFs={self},
                            localOrHDFSPath=pieceLocalOrHDFSPath,
                            columns=schema.names,
                            types=_cache.types,
                            nRows=None)

        self.__dict__.update(_cache)

        self._iCol = iCol
        self._tCol = tCol
        self.hasTS = iCol and tCol

        self._defaultMapper = defaultMapper

        self._reprSampleNPieces = min(reprSampleNPieces, self.nPieces)
        self._reprSampleSize = reprSampleSize

        self._minNonNullProportion = minNonNullProportion
        self._outlierTailProportion = outlierTailProportion
        self._maxNCats = maxNCats
        self._minProportionByMaxNCats = minProportionByMaxNCats

        self._emptyCache()

    # **********
    # IO METHODS
    # load / read
    # save

    @classmethod
    def load(cls, path, **kwargs):
        return cls(path=path, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        return cls(path=path, **kwargs)

    def save(self, *args, **kwargs):
        return NotImplemented

    # ***************
    # PYTHON STR/REPR
    # __repr__
    # __short_repr__

    @property
    def _pathRepr(self):
        return '"{}"'.format(self.path) \
            if isinstance(self.path, _STR_CLASSES) \
          else '{} Paths e.g. {}'.format(len(self.path), self.path[:3])

    def __repr__(self):
        cols_and_types_str = []

        if self._iCol:
            cols_and_types_str += ['(iCol) {}: {}'.format(self._iCol, self.type(self._iCol))]

        if self._tCol:
            cols_and_types_str += ['(tCol) {}: {}'.format(self._tCol, self.type(self._tCol))]

        cols_and_types_str += \
            ['{}: {}'.format(col, self.type(col))
             for col in self.contentCols]
        
        return '{:,}-piece {}{}[{}][{}]'.format(
            self.nPieces,
            '{:,}-row '.format(self._cache.nRows)
                if self._cache.nRows
                else ('approx-{:,.0f}-row '.format(self._cache.approxNRows)
                      if self._cache.approxNRows
                      else ''),
            type(self).__name__,
            self._pathRepr,
            ', '.join(cols_and_types_str))

    @property
    def __short_repr__(self):
        cols_desc_str = []

        if self._iCol:
            cols_desc_str += ['iCol: {}'.format(self._iCol)]

        if self._tCol:
            cols_desc_str += ['tCol: {}'.format(self._tCol)]

        cols_desc_str += ['{} content col(s)'.format(len(self.contentCols))]

        return '{:,}-piece {}{}[{}][{}]'.format(
            self.nPieces,
            '{:,}-row '.format(self._cache.nRows)
                if self._cache.nRows
                else ('approx-{:,.0f}-row '.format(self._cache.approxNRows)
                      if self._cache.approxNRows
                      else ''),
            type(self).__name__,
            self._pathRepr,
            ', '.join(cols_desc_str))

    # ***************
    # CACHING METHODS
    # _emptyCache
    # _inheritCache
    # pieceLocalOrHDFSPath

    def _emptyCache(self):
        self._cache = \
            _Namespace(
                reprSamplePiecePaths=None,
                reprSample=None,

                nRows=None,
                approxNRows=None,

                count={}, distinct={},   # approx.

                nonNullProportion={},   # approx.
                suffNonNullProportionThreshold={},
                suffNonNull={},

                sampleMin={}, sampleMax={}, sampleMean={}, sampleMedian={},
                outlierRstMin={}, outlierRstMax={}, outlierRstMean={}, outlierRstMedian={},

                colWidth={})

    def _inheritCache(self, arrowDF, *sameCols, **newColToOldColMappings):
        if arrowDF._cache.nRows:
            if self._cache.nRows is None:
                self._cache.nRows = arrowDF._cache.nRows
            else:
                assert self._cache.nRows == arrowDF._cache.nRows

        if arrowDF._cache.approxNRows and (self._cache.approxNRows is None):
            self._cache.approxNRows = arrowDF._cache.approxNRows

        commonCols = set(self.columns).intersection(ArrowDF.columns)

        if sameCols or newColToOldColMappings:
            for newCol, oldCol in newColToOldColMappings.items():
                assert newCol in self.columns
                assert oldCol in ArrowDF.columns

            for sameCol in commonCols.difference(newColToOldColMappings).intersection(sameCols):
                newColToOldColMappings[sameCol] = sameCol

        else:
            newColToOldColMappings = \
                {col: col
                 for col in commonCols}

        for cacheCategory in \
                ('count', 'distinct',
                 'nonNullProportion', 'suffNonNullProportionThreshold', 'suffNonNull',
                 'sampleMin', 'sampleMax', 'sampleMean', 'sampleMedian',
                 'outlierRstMin', 'outlierRstMax', 'outlierRstMean', 'outlierRstMedian',
                 'colWidth'):
            for newCol, oldCol in newColToOldColMappings.items():
                if oldCol in arrowDF._cache.__dict__[cacheCategory]:
                    self._cache.__dict__[cacheCategory][newCol] = \
                        arrowDF._cache.__dict__[cacheCategory][oldCol]

    def pieceLocalOrHDFSPath(self, piecePath):
        if (piecePath in self._PIECE_CACHES) and self._PIECE_CACHES[piecePath].localOrHDFSPath:
            return self._PIECE_CACHES[piecePath].localOrHDFSPath

        else:
            if piecePath.startswith('s3'):
                parsedURL = \
                    urlparse(
                        url=piecePath,
                        scheme='',
                        allow_fragments=True)

                localOrHDFSPath = \
                    os.path.join(
                        self._TMP_DIR_PATH,
                        parsedURL.netloc,
                        parsedURL.path[1:])

                fs.mkdir(
                    dir=os.path.dirname(localOrHDFSPath),
                    hdfs=False)

                self.s3Client.download_file(
                    Bucket=parsedURL.netloc,
                    Key=parsedURL.path[1:],
                    Filename=localOrHDFSPath)

            else:
                localOrHDFSPath = piecePath

            if piecePath in self._PIECE_CACHES:
                self._PIECE_CACHES[piecePath].localOrHDFSPath = localOrHDFSPath

            return localOrHDFSPath

    # ***********************
    # MAP-REDUCE (PARTITIONS)
    # _mr
    # collect

    def _mr(self, *piecePaths, **kwargs):
        _CHUNK_SIZE = 10 ** 5

        cols = kwargs.get('cols')
        if not cols:
            cols = None

        nSamplesPerPiece = kwargs.get('nSamplesPerPiece')

        organizeTS = kwargs.get('organizeTS', True)

        applyDefaultMapper = kwargs.get('applyDefaultMapper', True)

        mapper = kwargs.get('mapper')

        reducer = \
            kwargs.get(
                'reducer',
                lambda results:
                    pandas.concat(
                        objs=results,
                        axis='index',
                        join='outer',
                        join_axes=None,
                        ignore_index=True,
                        keys=None,
                        levels=None,
                        names=None,
                        verify_integrity=False,
                        copy=False))

        verbose = kwargs.pop('verbose', True)
        
        if not piecePaths:
            piecePaths = self.piecePaths

        results = []

        for piecePath in \
                (tqdm.tqdm(piecePaths)
                 if verbose
                 else piecePaths):
            pieceArrowTable = \
                read_table(
                    source=self.pieceLocalOrHDFSPath(piecePath=piecePath),
                    columns=cols,
                    nthreads=psutil.cpu_count(logical=True),
                    metadata=None,
                    use_pandas_metadata=False)

            pieceCache = self._PIECE_CACHES[piecePath]

            if not pieceCache.columns:
                pieceCache.columns = pieceArrowTable.schema.names
                self.columns.update(pieceCache.columns)

                for col in pieceCache.columns:
                    pieceCache.types[col] = _arrowType = \
                        pieceArrowTable.schema.field_by_name(col).type

                    if col in self.types:
                        assert _arrowType == self.types[col], \
                            '*** {} COLUMN {}: DETECTED TYPE {} != {} ***'.format(
                                piecePath, col, _arrowType, self.types[col])

                    else:
                        self.types[col] = _arrowType

            if pieceCache.nRows is None:
                pieceCache.nRows = pieceArrowTable.num_rows

            if nSamplesPerPiece and (nSamplesPerPiece < pieceCache.nRows):
                intermediateN = (nSamplesPerPiece * pieceCache.nRows) ** .5
                
                nChunks = int(math.ceil(pieceCache.nRows / _CHUNK_SIZE))
                nChunksForIntermediateN = int(math.ceil(intermediateN / _CHUNK_SIZE))

                nSamplesPerChunk = int(math.ceil(nSamplesPerPiece / nChunksForIntermediateN))

                if nChunksForIntermediateN < nChunks:
                    recordBatches = pieceArrowTable.to_batches(chunksize=_CHUNK_SIZE)

                    nRecordBatches = len(recordBatches)

                    assert nRecordBatches in (nChunks - 1, nChunks), \
                        '*** {}: {} vs. {} Record Batches ***'.format(piecePath, nRecordBatches, nChunks)

                    assert nChunksForIntermediateN <= nRecordBatches, \
                        '*** {}: {} vs. {} Record Batches ***'.format(piecePath, nChunksForIntermediateN, nRecordBatches)

                    chunkPandasDFs = []

                    for recordBatch in \
                            random.sample(
                                population=recordBatches,
                                k=nChunksForIntermediateN):
                        chunkPandasDF = recordBatch.to_pandas(nthreads=max(1, psutil.cpu_count() // 2))

                        for partitionKV in re.findall('[^/]+=[^/]+/', piecePath):
                            k, v = partitionKV.split('=')

                            chunkPandasDF[str(k)] = \
                                datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date() \
                                if k == DATE_COL \
                                else v[:-1]

                        if organizeTS and self._tCol:
                            assert self._tCol in chunkPandasDF.columns, \
                                '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self._tCol, chunkPandasDF.columns)

                        if self._iCol:
                            assert self._iCol in chunkPandasDF.columns, \
                                '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self._iCol, chunkPandasDF.columns)

                            try:
                                chunkPandasDF = \
                                    gen_aux_cols(
                                        df=chunkPandasDF.loc[
                                            pandas.notnull(chunkPandasDF[self._iCol]) &
                                            pandas.notnull(chunkPandasDF[self._tCol])],
                                        i_col=self._iCol, t_col=self._tCol)

                            except Exception as err:
                                print('*** {} ***'.format(piecePath))
                                raise err

                        else:
                            try:
                                chunkPandasDF = \
                                    gen_aux_cols(
                                        df=chunkPandasDF.loc[pandas.notnull(chunkPandasDF[self._tCol])],
                                        i_col=None, t_col=self._tCol)

                            except Exception as err:
                                print('*** {} ***'.format(piecePath))
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
                                        # Default None results in equal probability weighting.
                                        # If passed a Series, will align with target object on index.
                                        # Index values in weights not found in sampled object will be ignored
                                        # and index values in sampled object not in weights will be assigned weights of zero.
                                        # If called on a DataFrame, will accept the name of a column when axis = 0.
                                        # Unless weights are a Series, weights must be same length as axis being sampled.
                                        # If weights do not sum to 1, they will be normalized to sum to 1.
                                        # Missing values in the weights column will be treated as zero.
                                        # inf and -inf values not allowed.
                                    random_state=None,
                                        # Seed for the random number generator (if int), or numpy RandomState object.
                                    axis='index')

                        chunkPandasDFs.append(chunkPandasDF)

                    piecePandasDF = \
                        pandas.concat(
                            objs=chunkPandasDFs,
                            axis='index',
                            join='outer',
                            join_axes=None,
                            ignore_index=True,
                            keys=None,
                            levels=None,
                            names=None,
                            verify_integrity=False,
                            copy=False)

                else:
                    piecePandasDF = \
                        pieceArrowTable.to_pandas(
                            nthreads=max(1, psutil.cpu_count(logical=True) // 2),
                                # For the default, we divide the CPU count by 2
                                # because most modern computers have hyperthreading turned on,
                                # so doubling the CPU count beyond the number of physical cores does not help
                            strings_to_categorical=False,
                            memory_pool=None,
                            zero_copy_only=None,
                            categories=[],
                            integer_object_nulls=False)

                    for partitionKV in re.findall('[^/]+=[^/]+/', piecePath):
                        k, v = partitionKV.split('=')

                        piecePandasDF[str(k)] = \
                            datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date() \
                            if k == DATE_COL \
                            else v[:-1]

                    if organizeTS and self._tCol:
                        assert self._tCol in piecePandasDF.columns, \
                            '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self._tCol, piecePandasDF.columns)

                        if self._iCol:
                            assert self._iCol in piecePandasDF.columns, \
                                '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self._iCol, piecePandasDF.columns)

                            try:
                                piecePandasDF = \
                                    gen_aux_cols(
                                        df=piecePandasDF.loc[
                                            pandas.notnull(piecePandasDF[self._iCol]) &
                                            pandas.notnull(piecePandasDF[self._tCol])],
                                        i_col=self._iCol, t_col=self._tCol)

                            except Exception as err:
                                print('*** {} ***'.format(piecePath))
                                raise err

                        else:
                            try:
                                piecePandasDF = \
                                    gen_aux_cols(
                                        df=piecePandasDF.loc[pandas.notnull(piecePandasDF[self._tCol])],
                                        i_col=None, t_col=self._tCol)

                            except Exception as err:
                                print('*** {} ***'.format(piecePath))
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
                                # Default None results in equal probability weighting.
                                # If passed a Series, will align with target object on index.
                                # Index values in weights not found in sampled object will be ignored
                                # and index values in sampled object not in weights will be assigned weights of zero.
                                # If called on a DataFrame, will accept the name of a column when axis = 0.
                                # Unless weights are a Series, weights must be same length as axis being sampled.
                                # If weights do not sum to 1, they will be normalized to sum to 1.
                                # Missing values in the weights column will be treated as zero.
                                # inf and -inf values not allowed.
                            random_state=None,
                                # Seed for the random number generator (if int), or numpy RandomState object.
                            axis='index')

            else:
                piecePandasDF = \
                    pieceArrowTable.to_pandas(
                        nthreads=max(1, psutil.cpu_count(logical=True) // 2),
                            # For the default, we divide the CPU count by 2
                            # because most modern computers have hyperthreading turned on,
                            # so doubling the CPU count beyond the number of physical cores does not help
                        strings_to_categorical=False,
                        memory_pool=None,
                        zero_copy_only=None,
                        categories=[],
                        integer_object_nulls=False)

                for partitionKV in re.findall('[^/]+=[^/]+/', piecePath):
                    k, v = partitionKV.split('=')

                    piecePandasDF[str(k)] = \
                        datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date() \
                        if k == DATE_COL \
                        else v[:-1]

                if organizeTS and self._tCol:
                    assert self._tCol in piecePandasDF.columns, \
                        '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self._tCol, piecePandasDF.columns)

                    if self._iCol:
                        assert self._iCol in piecePandasDF.columns, \
                            '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self._iCol, piecePandasDF.columns)

                        try:
                            piecePandasDF = \
                                gen_aux_cols(
                                    df=piecePandasDF.loc[
                                        pandas.notnull(piecePandasDF[self._iCol]) &
                                        pandas.notnull(piecePandasDF[self._tCol])],
                                    i_col=self._iCol, t_col=self._tCol)

                        except Exception as err:
                            print('*** {} ***'.format(piecePath))
                            raise err

                    else:
                        try:
                            piecePandasDF = \
                                gen_aux_cols(
                                    df=piecePandasDF.loc[pandas.notnull(piecePandasDF[self._tCol])],
                                    i_col=None, t_col=self._tCol)

                        except Exception as err:
                            print('*** {} ***'.format(piecePath))
                            raise err

            if applyDefaultMapper and self._defaultMapper:
                piecePandasDF = self._defaultMapper(piecePandasDF)

            results.append(
                mapper(piecePandasDF)
                if mapper
                else piecePandasDF)

        return reducer(results)

    def collect(self, *cols, **kwargs):
        return self._mr(cols=cols if cols else None, **kwargs)

    # *************************
    # KEY (SETTABLE) PROPERTIES
    # iCol
    # tCol
    # defaultMapper

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

    @property
    def defaultMapper(self):
        return self._defaultMapper

    @defaultMapper.setter
    def defaultMapper(self, defaultMapper):
        self._defaultMapper = defaultMapper

    @defaultMapper.deleter
    def defaultMapper(self):
        self._defaultMapper = None

    # ***********
    # REPR SAMPLE
    # reprSamplePiecePaths
    # _assignReprSample
    # __getattr__
    # __getitem__

    @property
    def reprSamplePiecePaths(self):
        if self._cache.reprSamplePiecePaths is None:
            self._cache.reprSamplePiecePaths = \
                random.sample(
                    population=self.piecePaths,
                    k=self._reprSampleNPieces)

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

    # *** DISABLED: __getattr__ is generally bug-prone ***
    # def __getattr__(self, attr):
    #     return getattr(self.reprSample, attr)

    def __getitem__(self, item):
        return self.reprSample[item]

    # *********************
    # ROWS, COLUMNS & TYPES
    # nRows
    # approxNRows
    # type / typeIsNum / typeIsComplex

    @property
    def nRows(self):
        if self._cache.nRows is None:
            self._cache.nRows = \
                sum(read_metadata(where=self.pieceLocalOrHDFSPath(piecePath=piecePath)).num_rows
                    for piecePath in tqdm.tqdm(self.piecePaths))

        return self._cache.nRows

    @property
    def approxNRows(self):
        if self._cache.approxNRows is None:
            self._cache.approxNRows = \
                self.nPieces \
                * sum(read_metadata(where=self.pieceLocalOrHDFSPath(piecePath=piecePath)).num_rows
                      for piecePath in tqdm.tqdm(self.reprSamplePiecePaths)) \
                / self._reprSampleNPieces

        return self._cache.approxNRows

    def type(self, col):
        return self.types[col]

    def typeIsNum(self, col):
        return is_num(self.type(col))

    def typeIsComplex(self, col):
        return is_complex(self.type(col))

    # *************
    # COLUMN GROUPS
    # indexCols
    # tRelAuxCols
    # possibleFeatureContentCols
    # possibleCatContentCols

    @property
    def indexCols(self):
        return ((self._iCol,) if self._iCol else ()) \
             + ((self._tCol,) if self._tCol else ())

    @property
    def tRelAuxCols(self):
        return (self._T_ORD_COL, self._T_DELTA_COL) \
            if self.hasTS \
          else ()

    @property
    def possibleFeatureContentCols(self):
        chk = lambda t: is_boolean(t) or is_string(t) or is_num(t)

        return tuple(
            col for col in self.contentCols
                if chk(self.type(col)))

    @property
    def possibleCatContentCols(self):
        return tuple(
            col for col in self.contentCols
                if is_possible_cat(self.type(col)))

    def sample(self, *cols, **kwargs):
        n = kwargs.pop('n', 1)

        piecePaths = kwargs.pop('piecePaths')

        verbose = kwargs.pop('verbose', True)

        if piecePaths:
            nSamplePieces = len(piecePaths)

        else:
            minNPieces = kwargs.pop('minNPieces', self._reprSampleNPieces)
            maxNPieces = kwargs.pop('maxNPieces', None)

            nSamplePieces = \
                max(int(math.ceil(((min(n, self.nRows) / self.nRows) ** .5)
                                  * self.nPieces)),
                    minNPieces)

            if maxNPieces:
                nSamplePieces = min(nSamplePieces, maxNPieces)

            if nSamplePieces < self.nPieces:
                piecePaths = \
                    random.sample(
                        population=self.piecePaths,
                        k=nSamplePieces)

            else:
                nSamplePieces = self.nPieces
                piecePaths = self.piecePaths

        return self._mr(
            *piecePaths,
            cols=cols,
            nSamplesPerPiece=int(math.ceil(n / nSamplePieces)),
            organizeTS=True,
            applyDefaultMapper=True,
            verbose=verbose)

    # ****************
    # COLUMN PROFILING
    # _nonNullCol
    # count
    # nonNullProportion
    # suffNonNull
    # approxQuantile
    # sampleStat / sampleMedian
    # outlierRstStat / outlierRstMin / outlierRstMax / outlierRstMedian
    # profile

    def _nonNullCol(self, col, lower=None, upper=None, strict=False):
        colType = self.type(col)

        condition = \
            '({} IS NOT NULL)'.format(col) + \
            ('' if colType.startswith(_ARRAY_TYPE_PREFIX) or
                   colType.startswith(_MAP_TYPE_PREFIX) or
                   colType.startswith(_STRUCT_TYPE_PREFIX)
             else " AND (STRING({}) != 'NaN')".format(col))

        if (colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _NUM_TYPES)) and \
                (pandas.notnull(lower) or pandas.notnull(upper)):
            _debugLogCondition = False

            if colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _FLOAT_TYPES):
                numStrFormatter = '%.9f'

                if pandas.notnull(lower) and pandas.notnull(upper) and (lower + 1e-6 > upper):
                    self.stdout_logger.warning(
                        msg='*** LOWER {} >= UPPER {} ***'
                            .format(lower, upper))

                    upper = lower + 1e-6

                    _debugLogCondition = True

            else:
                numStrFormatter = '%i'

            equalSignStr = '' if strict else '='

            condition += ' AND {}{}{}'.format(
                '' if pandas.isnull(lower)
                else '({} >{} {})'.format(col, equalSignStr, numStrFormatter % lower),

                '' if pandas.isnull(lower) or pandas.isnull(upper)
                else ' AND ',

                '' if pandas.isnull(upper)
                else '({} <{} {})'.format(col, equalSignStr, numStrFormatter % upper))

            if _debugLogCondition:
                self.stdout_logger.debug(
                    msg='*** CONDITION: "{}" ***'
                        .format(condition))

        return self._sparkDF[[col]].filter(condition=condition)

    @_docstr_verbose
    def count(self, *cols, **kwargs):
        """
        Return:
            - If 1 column name is given, return its corresponding non-``NULL`` count

            - If multiple column names are given, return a {``col``: corresponding non-``NULL`` count} *dict*

            - If no column names are given, return a {``col``: corresponding non-``NULL`` count} *dict* for all columns

        Args:
             *cols (str): column name(s)

             **kwargs:
        """
        if not cols:
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**
                             {col: self.count(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            if col not in self._cache.count:
                verbose = True \
                    if arimo.debug.ON \
                    else kwargs.get('verbose')

                if verbose:
                    tic = time.time()

                self._cache.count[col] = result = \
                    self._sparkDF.select(sparkSQLFuncs.count(col)).first()[0] \
                        if self.typeIsComplex(col) \
                        else self._nonNullCol(col=col).count()

                assert isinstance(result, int), \
                    '*** "{}" COUNT = {} ***'.format(col, result)

                if verbose:
                    toc = time.time()
                    self.stdout_logger.info(
                        msg='No. of Non-NULLs of Column "{}" = {:,}   <{:,.1f} s>'
                            .format(col, result, toc - tic))

            return self._cache.count[col]

    @_docstr_verbose
    def nonNullProportion(self, *cols, **kwargs):
        """
        Return:
            - If 1 column name is given, return its *approximate* non-``NULL`` proportion

            - If multiple column names are given, return {``col``: approximate non-``NULL`` proportion} *dict*

            - If no column names are given, return {``col``: approximate non-``NULL`` proportion} *dict* for all columns

        Args:
             *cols (str): column name(s)

             **kwargs:
        """
        if not cols:
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**
                             {col: self.nonNullProportion(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            if col not in self._cache.nonNullProportion:
                self._cache.nonNullProportion[col] = \
                    self.reprSample.count(col, **kwargs) / self.reprSampleSize

            return self._cache.nonNullProportion[col]

    @_docstr_verbose
    def suffNonNull(self, *cols, **kwargs):
        """
        Check whether the columns has at least ``.minNonNullProportion`` of non-``NULL`` values

        Return:
            - If 1 column name is given, return ``True``/``False``

            - If multiple column names are given, return a {``col``: ``True`` or ``False``} *dict*

            - If no column names are given, return a {``col``: ``True`` or ``False``} *dict* for all columns

        Args:
            *cols (str): column names

            **kwargs:
        """
        if not cols:
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**
                             {col: self.suffNonNull(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            minNonNullProportion = self._minNonNullProportion[col]

            outdatedSuffNonNullProportionThreshold = False

            if col in self._cache.suffNonNullProportionThreshold:
                if self._cache.suffNonNullProportionThreshold[col] != minNonNullProportion:
                    outdatedSuffNonNullProportionThreshold = True
                    self._cache.suffNonNullProportionThreshold[col] = minNonNullProportion

            else:
                self._cache.suffNonNullProportionThreshold[col] = minNonNullProportion

            if (col not in self._cache.suffNonNull) or outdatedSuffNonNullProportionThreshold:
                self._cache.suffNonNull[col] = \
                    self.nonNullProportion(col) >= self._cache.suffNonNullProportionThreshold[col]

            return self._cache.suffNonNull[col]

    @_docstr_verbose
    def distinct(self, col=None, count=True, collect=True, **kwargs):
        """
        Return:
            *Approximate* list of distinct values of ``ADF``'s column ``col``,
                with optional descending-sorted counts for those values

        Args:
            col (str): name of a column

            count (bool): whether to count the number of appearances of each distinct value of the specified ``col``

            collect (bool): whether to return a ``pandas.DataFrame`` (``collect=True``) or a ``Spark SQL DataFrame``

            **kwargs:
        """
        if col:
            if col in self._cache.distinct:
                df = self._cache.distinct[col]
                assert isinstance(df, pandas.DataFrame)
                if len(df.columns) or not count:
                    return df

            verbose = True \
                if arimo.debug.ON \
                else kwargs.get('verbose')

            if verbose:
                msg = 'Profiling Distinct Values of Column "{}"...'.format(col)
                self.stdout_logger.info(msg)
                tic = time.time()

            adf = self.reprSample(
                'SELECT \
                    {0}, \
                    COUNT(*) AS __sample_count__, \
                    (COUNT(*) / {1}) AS __proportion__ \
                FROM \
                    this \
                GROUP BY \
                    {0} \
                ORDER BY \
                    __sample_count__ DESC'
                    .format(col, self.reprSampleSize),
                **kwargs) \
                if count \
                else \
                self.reprSample(
                    'SELECT \
                        DISTINCT({}) \
                    FROM \
                        this'.format(col),
                    **kwargs)

            if collect:
                df = adf.toPandas()

                dups = {k: v
                        for k, v in Counter(df[col]).items()
                        if v > 1}

                if dups:
                    assert all(pandas.isnull(k) for k in dups), \
                        '*** {}.distinct("{}"): POSSIBLE SPARK SQL/HIVEQL BUG: DUPLICATES {} ***'.format(self, col, dups)

                    index_of_first_row_with_null = None
                    row_indices_to_delete = []

                    for i, row in df.iterrows():
                        if pandas.isnull(row[col]):
                            if index_of_first_row_with_null is None:
                                index_of_first_row_with_null = i

                            else:
                                row_indices_to_delete.append(i)

                                if count:
                                    df.at[index_of_first_row_with_null, '__sample_count__'] += df.at[i, '__sample_count__']
                                    df.at[index_of_first_row_with_null, '__proportion__'] += df.at[i, '__proportion__']

                    df.drop(
                        index=row_indices_to_delete,
                        level=None,
                        inplace=True,
                        errors='raise')

                    if count:
                        df.sort_values(
                            by='__sample_count__',
                            ascending=False,
                            inplace=True,
                            kind='quicksort',
                            na_position='last')

                        df.reset_index(
                            level=None,
                            drop=True,
                            inplace=True,
                            col_level=0,
                            col_fill='')

                self._cache.distinct[col] = \
                    result = \
                    df.set_index(
                        keys=col,
                        drop=True,
                        append=False,
                        inplace=False,
                        verify_integrity=False)

            else:
                result = adf

            if verbose:
                toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

            return result

        else:
            return self._decorate(
                obj=self._sparkDF.distinct(),
                nRows=None,
                **kwargs)

    @lru_cache()
    def approxQuantile(self, *cols, **kwargs):   # make Spark SQL approxQuantile method NULL-resistant
        if len(cols) > 1:
            return Namespace(**
                             {col: self.approxQuantile(col, **kwargs)
                              for col in cols})

        elif len(cols) == 1:
            col = cols[0]

            prob = kwargs.get('probabilities', .5)
            _multiProbs = isinstance(prob, (list, tuple))

            relErr = kwargs.get('relativeError', 0)

            if self.count(col):
                result = \
                    self._nonNullCol(col=col) \
                        .approxQuantile(
                        col=col,
                        probabilities=prob
                        if _multiProbs
                        else (prob,),
                        relativeError=relErr)

                return result \
                    if _multiProbs \
                    else result[0]

            else:
                return len(prob) * [numpy.nan] \
                    if _multiProbs \
                    else numpy.nan

    @_docstr_verbose
    def sampleStat(self, *cols, **kwargs):
        """
        *Approximate* measurements of a certain statistic on **numerical** columns

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
            cols = [col for col in self.contentCols
                    if self.typeIsNum(col)]

        if len(cols) > 1:
            return Namespace(**
                             {col: self.sampleStat(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                stat = kwargs.pop('stat', 'mean').lower()
                if stat == 'avg':
                    stat = 'mean'
                capitalizedStatName = stat.capitalize()
                s = 'sample{}'.format(capitalizedStatName)

                if hasattr(self, s):
                    return getattr(self, s)(col, **kwargs)

                else:
                    if s not in self._cache:
                        setattr(self._cache, s, {})
                    cache = getattr(self._cache, s)

                    if col not in cache:
                        verbose = True \
                            if arimo.debug.ON \
                            else kwargs.get('verbose')

                        if verbose:
                            tic = time.time()

                        cache[col] = result = \
                            self.reprSample \
                                ._nonNullCol(col=col) \
                                .select(getattr(sparkSQLFuncs, stat)(col)) \
                                .first()[0]

                        assert isinstance(result, (float, int)), \
                            '*** "{}" SAMPLE {} = {} ***'.format(col, capitalizedStatName.upper(), result)

                        if verbose:
                            toc = time.time()
                            self.stdout_logger.info(
                                msg='Sample {} for Column "{}" = {:,.3g}   <{:,.1f} s>'
                                    .format(capitalizedStatName, col, result, toc - tic))

                    return cache[col]

    def sampleMedian(self, *cols, **kwargs):
        if not cols:
            cols = [col for col in self.contentCols
                    if self.typeIsNum(col)]

        if len(cols) > 1:
            return Namespace(**
                             {col: self.sampleMedian(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                if 'sampleMedian' not in self._cache:
                    self._cache.sampleMedian = {}

                if col not in self._cache.sampleMedian:
                    verbose = True \
                        if arimo.debug.ON \
                        else kwargs.get('verbose')

                    if verbose:
                        tic = time.time()

                    self._cache.sampleMedian[col] = result = \
                        self.reprSample \
                            .approxQuantile(
                            col,
                            probabilities=.5,
                            relativeError=0)

                    assert isinstance(result, (float, int)), \
                        '*** "{}" SAMPLE MEDIAN = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Sample Median of Column "{}" = {:,.3g}   <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.sampleMedian[col]

    def outlierRstStat(self, *cols, **kwargs):
        if not cols:
            cols = [col for col in self.contentCols
                    if self.typeIsNum(col)]

        if len(cols) > 1:
            return Namespace(**
                             {col: self.outlierRstStat(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                stat = kwargs.pop('stat', 'mean').lower()
                if stat == 'avg':
                    stat = 'mean'
                capitalizedStatName = stat.capitalize()
                s = 'outlierRst{}'.format(capitalizedStatName)

                if hasattr(self, s):
                    return getattr(self, s)(col, **kwargs)

                else:
                    if s not in self._cache:
                        setattr(self._cache, s, {})
                    cache = getattr(self._cache, s)

                    if col not in cache:
                        verbose = True \
                            if arimo.debug.ON \
                            else kwargs.get('verbose')

                        if verbose:
                            tic = time.time()

                        outlierTails = kwargs.pop('outlierTails', 'both')

                        cache[col] = result = \
                            self.reprSample \
                                ._nonNullCol(
                                col=col,
                                lower=self.outlierRstMin(col)
                                if outlierTails in ('lower', 'both')
                                else None,
                                upper=self.outlierRstMax(col)
                                if outlierTails in ('upper', 'both')
                                else None,
                                strict=False) \
                                .select(getattr(sparkSQLFuncs, stat)(col)) \
                                .first()[0]

                        if result is None:
                            self.stdout_logger.warning(
                                msg='*** "{}" OUTLIER-RESISTANT {} = {} ***'.format(col, capitalizedStatName.upper(), result))

                            result = self.outlierRstMin(col)

                        assert isinstance(result, (float, int))

                        if verbose:
                            toc = time.time()
                            self.stdout_logger.info(
                                msg='Outlier-Resistant {} for Column "{}" = {:,.3g}   <{:,.1f} s>'
                                    .format(capitalizedStatName, col, result, toc - tic))

                    return cache[col]

    def outlierRstMin(self, *cols, **kwargs):
        if not cols:
            cols = [col for col in self.contentCols
                    if self.typeIsNum(col)]

        if len(cols) > 1:
            return Namespace(**
                             {col: self.outlierRstMin(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]
            if self.typeIsNum(col):
                if 'outlierRstMin' not in self._cache:
                    self._cache.outlierRstMin = {}

                if col not in self._cache.outlierRstMin:
                    verbose = True \
                        if arimo.debug.ON \
                        else kwargs.get('verbose')

                    if verbose:
                        tic = time.time()

                    outlierRstMin = \
                        self.reprSample \
                            .approxQuantile(
                            col,
                            probabilities=self._outlierTailProportion[col],
                            relativeError=0)

                    sampleMin = self.sampleStat(col, stat='min')
                    sampleMedian = self.sampleMedian(col)

                    self._cache.outlierRstMin[col] = result = \
                        self.reprSample \
                            ._nonNullCol(
                            col=col,
                            lower=sampleMin,
                            strict=True) \
                            .select(sparkSQLFuncs.min(col)) \
                            .first()[0] \
                            if (outlierRstMin == sampleMin) and (outlierRstMin < sampleMedian) \
                            else outlierRstMin

                    assert isinstance(result, (float, int)), \
                        '*** "{}" OUTLIER-RESISTANT MIN = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Outlier-Resistant Min of Column "{}" = {:,.3g}   <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.outlierRstMin[col]

    def outlierRstMax(self, *cols, **kwargs):
        if not cols:
            cols = [col for col in self.contentCols
                    if self.typeIsNum(col)]

        if len(cols) > 1:
            return Namespace(**
                             {col: self.outlierRstMax(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]
            if self.typeIsNum(col):
                if 'outlierRstMax' not in self._cache:
                    self._cache.outlierRstMax = {}

                if col not in self._cache.outlierRstMax:
                    verbose = True \
                        if arimo.debug.ON \
                        else kwargs.get('verbose')

                    if verbose:
                        tic = time.time()

                    outlierRstMax = \
                        self.reprSample \
                            .approxQuantile(
                            col,
                            probabilities=1 - self._outlierTailProportion[col],
                            relativeError=0)

                    sampleMax = self.sampleStat(col, stat='max')
                    sampleMedian = self.sampleMedian(col)

                    self._cache.outlierRstMax[col] = result = \
                        self.reprSample \
                            ._nonNullCol(
                            col=col,
                            upper=sampleMax,
                            strict=True) \
                            .select(sparkSQLFuncs.max(col)) \
                            .first()[0] \
                            if (outlierRstMax == sampleMax) and (outlierRstMax > sampleMedian) \
                            else outlierRstMax

                    assert isinstance(result, (float, int)), \
                        '*** "{}" OUTLIER-RESISTANT MAX = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Outlier-Resistant Max of Column "{}" = {:,.3g}   <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.outlierRstMax[col]

    def outlierRstMedian(self, *cols, **kwargs):
        if not cols:
            cols = [col for col in self.contentCols
                    if self.typeIsNum(col)]

        if len(cols) > 1:
            return Namespace(**
                             {col: self.outlierRstMedian(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]
            if self.typeIsNum(col):
                if 'outlierRstMedian' not in self._cache:
                    self._cache.outlierRstMedian = {}

                if col not in self._cache.outlierRstMedian:
                    verbose = kwargs.get('verbose')
                    if verbose:
                        tic = time.time()

                    self._cache.outlierRstMedian[col] = result = \
                        self.reprSample \
                            ._nonNullCol(
                            col=col,
                            lower=self.outlierRstMin(col),
                            upper=self.outlierRstMax(col),
                            strict=False) \
                            .approxQuantile(
                            col=col,
                            probabilities=(.5,),
                            relativeError=0)[0]

                    assert isinstance(result, (float, int)), \
                        '*** "{}" OUTLIER-RESISTANT MEDIAN = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Outlier-Resistant Median of Column "{}" = {:,.3g}    <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.outlierRstMedian[col]

    @_docstr_verbose
    def profile(self, *cols, **kwargs):
        """
        Return:
            *dict* of profile of salient statistics on specified columns of ``ADF``

        Args:
            *cols (str): names of column(s) to profile

            **kwargs:

                - **profileCat** *(bool, default = True)*: whether to profile possible categorical columns

                - **profileNum** *(bool, default = True)*: whether to profile numerical columns

                - **skipIfInvalid** *(bool, default = False)*: whether to skip profiling if column does not have enough non-NULLs
        """
        if not cols:
            cols = self.contentCols

        asDict = kwargs.pop('asDict', False)

        if len(cols) > 1:
            return Namespace(**
                             {col: self.profile(col, **kwargs)
                              for col in cols})

        else:
            col = cols[0]

            verbose = True \
                if arimo.debug.ON \
                else kwargs.get('verbose')

            if verbose:
                msg = 'Profiling Column "{}"...'.format(col)
                self.stdout_logger.info(msg)
                tic = time.time()

            colType = self.type(col)
            profile = Namespace(type=colType)

            # non-NULL Proportions
            profile.nonNullProportion = \
                self.nonNullProportion(
                    col,
                    verbose=verbose > 1)

            if self.suffNonNull(col) or (not kwargs.get('skipIfInvalid', False)):
                # profile categorical column
                if kwargs.get('profileCat', True) and (colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _POSSIBLE_CAT_TYPES)):
                    profile.distinctProportions = \
                        self.distinct(
                            col=col,
                            count=True,
                            collect=True,
                            verbose=verbose > 1).__proportion__

                # profile numerical column
                if kwargs.get('profileNum', True) and (colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _NUM_TYPES)):
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
                        quantilesOfInterest[outlierTailProportion] = outlierRstMin
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
                        quantilesOfInterest[1 - outlierTailProportion] = outlierRstMax
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

                    if quantileProbsToQuery:
                        quantilesOfInterest[numpy.isnan(quantilesOfInterest)] = \
                            self.reprSample \
                                .approxQuantile(
                                col,
                                probabilities=quantileProbsToQuery,
                                relativeError=0)

                    sampleMin, outlierRstMin, sampleMedian, outlierRstMax, sampleMax = quantilesOfInterest

                    if toCacheSampleMin:
                        self._cache.sampleMin[col] = sampleMin

                    if toCacheOutlierRstMin:
                        if (outlierRstMin == sampleMin) and (outlierRstMin < sampleMedian):
                            outlierRstMin = \
                                self.reprSample \
                                    ._nonNullCol(
                                    col,
                                    lower=sampleMin,
                                    strict=True) \
                                    .select(sparkSQLFuncs.min(col)) \
                                    .first()[0]
                        self._cache.outlierRstMin[col] = outlierRstMin

                    if toCacheSampleMedian:
                        self._cache.sampleMedian[col] = sampleMedian

                    if toCacheOutlierRstMax:
                        if (outlierRstMax == sampleMax) and (outlierRstMax > sampleMedian):
                            outlierRstMax = \
                                self.reprSample \
                                    ._nonNullCol(
                                    col,
                                    upper=sampleMax,
                                    strict=True) \
                                    .select(sparkSQLFuncs.max(col)) \
                                    .first()[0]
                        self._cache.outlierRstMax[col] = outlierRstMax

                    if toCacheSampleMax:
                        self._cache.sampleMax[col] = sampleMax

                    profile.sampleRange = sampleMin, sampleMax
                    profile.outlierRstRange = outlierRstMin, outlierRstMax

                    profile.sampleMean = \
                        self.sampleStat(
                            col,
                            verbose=verbose)

                    profile.outlierRstMean = \
                        self._cache.outlierRstMean.get(
                            col,
                            self.outlierRstStat(
                                col,
                                verbose=verbose))

                    profile.outlierRstMedian = \
                        self._cache.outlierRstMedian.get(
                            col,
                            self.outlierRstMedian(
                                col,
                                verbose=verbose))

            if verbose:
                toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

            return Namespace(**{col: profile}) \
                if asDict \
                else profile

    # *********
    # DATA PREP
    # fillna
    # prep

    @_docstr_verbose
    def fillna(self, *cols, **kwargs):
        """
        Fill/interpolate ``NULL``/``NaN`` values

        Return:
            ``ADF`` with ``NULL``/``NaN`` values filled/interpolated

        Args:
            *args (str): names of column(s) to fill/interpolate

            **kwargs:

                - **method** *(str)*: one of the following methods to fill ``NULL`` values in **numerical** columns,
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

                    (*NOTE:* for an ``ADF`` with a ``.tCol`` set, ``NumPy/Pandas NaN`` values cannot be filled;
                        it is best that such *Python* values be cleaned up before they get into Spark)

                - **value**: single value, or *dict* of values by column name,
                    to use if ``method`` is ``None`` or not applicable

                - **outlierTails** *(str or dict of str, default = 'both')*: specification of in which distribution tail (``None``, ``lower``, ``upper`` and ``both`` (default)) of each numerical column out-lying values may exist

                - **fillOutliers** *(bool or list of column names, default = False)*: whether to treat detected out-lying values as ``NULL`` values to be replaced in the same way

                - **loadPath** *(str)*: HDFS path to load existing ``NULL``-filling data transformations

                - **savePath** *(str)*: HDFS path to save new ``NULL``-filling data transformations
        """
        _TS_FILL_METHODS = \
            'avg_partition', 'mean_partition', 'min_partition', 'max_partition', \
            'avg_before', 'mean_before', 'min_before', 'max_before', \
            'avg_after', 'mean_after', 'min_after', 'max_after'

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
                    partition=
                    '{} AS (PARTITION BY {}, {})'
                        .format(_TS_WINDOW_NAMES.partition, self._iCol, self._T_CHUNK_COL),
                    before=
                    '{} AS (PARTITION BY {}, {} ORDER BY {} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)'
                        .format(_TS_WINDOW_NAMES.before, self._iCol, self._T_CHUNK_COL, self._T_ORD_COL),
                    after=
                    '{} AS (PARTITION BY {}, {} ORDER BY {} ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)'
                        .format(_TS_WINDOW_NAMES.after, self._iCol, self._T_CHUNK_COL, self._T_ORD_COL))

        returnDetails = kwargs.pop('returnDetails', False)
        returnSQLTransformer = kwargs.pop('returnSQLTransformer', False)
        loadPath = kwargs.pop('loadPath', None)
        savePath = kwargs.pop('savePath', None)

        verbose = kwargs.pop('verbose', False)
        if arimo.debug.ON:
            verbose = True

        if loadPath:
            if verbose:
                message = 'Loading NULL-Filling SQL Transformations from HDFS Paths {}...'.format(loadPath)
                self.stdout_logger.info(message)
                tic = time.time()

            sqlTransformer = \
                SQLTransformer.load(
                    path=loadPath)

            details = None

        else:
            value = kwargs.pop('value', None)

            method = kwargs.pop(
                'method',
                'mean' if value is None
                else None)

            cols = set(cols)

            if isinstance(method, dict):
                cols.update(method)

            if isinstance(value, dict):
                cols.update(value)

            if not cols:
                cols = set(self.contentCols)

            cols.difference_update(
                self.indexCols +
                (self._T_ORD_COL, self._T_ORD_IN_CHUNK_COL))

            nulls = kwargs.pop('nulls', {})

            for col in cols:
                if col in nulls:
                    colNulls = nulls[col]
                    assert isinstance(colNulls, (list, tuple)) and (len(colNulls) == 2) \
                           and ((colNulls[0] is None) or isinstance(colNulls[0], (float, int))) \
                           and ((colNulls[1] is None) or isinstance(colNulls[1], (float, int)))

                else:
                    nulls[col] = (None, None)

            outlierTails = kwargs.pop('outlierTails', {})
            if isinstance(outlierTails, _STR_CLASSES):
                outlierTails = \
                    {col: outlierTails
                     for col in cols}

            fillOutliers = kwargs.pop('fillOutliers', False)
            fillOutliers = \
                cols \
                    if fillOutliers is True \
                    else to_iterable(fillOutliers)

            tsWindowDefs = set()
            details = {}

            if verbose:
                message = 'NULL-Filling Columns {}...'.format(
                    ', '.join('"{}"'.format(col) for col in cols))
                self.stdout_logger.info(message)
                tic = time.time()

            for col in cols:
                colType = self.type(col)
                colFallBackVal = None

                if colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _NUM_TYPES):
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
                                "NULL-Filling Methods {} Not Supported for Non-Time-Series ADFs".format(
                                    ', '.join(s.upper() for s in _TS_FILL_METHODS))

                            methodForCol, window = methodForCol

                        else:
                            methodForCol = methodForCol[0]

                            if self.hasTS:
                                window = None

                        colFallBackVal = \
                            self.outlierRstStat(
                                col,
                                stat=methodForCol
                                if (not self.hasTS) or (window is None) or (window == 'partition')
                                else 'mean',
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
                    valFormatter = \
                        '%f' if colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _FLOAT_TYPES) \
                            else ('%i' if colType in _INT_TYPES
                                  else ("'%s'" if (colType == _STR_TYPE) and
                                                  isinstance(colFallBackVal, _STR_CLASSES)
                                        else '%s'))

                    fallbackStrs = [valFormatter % colFallBackVal]

                    lowerNull, upperNull = colNulls = nulls[col]

                    if isNum and self.hasTS and window:
                        partitionFallBackStrTemplate = \
                            "%s(CASE WHEN (STRING(%s) = 'NaN')%s%s%s%s THEN NULL ELSE %s END) OVER %s"

                        fallbackStrs.insert(0,
                                            partitionFallBackStrTemplate
                                            % (methodForCol,
                                               col,
                                               '' if lowerNull is None
                                               else ' OR ({} <= {})'.format(col, lowerNull),
                                               '' if upperNull is None
                                               else ' OR ({} >= {})'.format(col, upperNull),
                                               ' OR (%s < %s)' % (col, valFormatter % self.outlierRstMin(col))
                                               if fixLowerTail
                                               else '',
                                               ' OR (%s > %s)' % (col, valFormatter % self.outlierRstMax(col))
                                               if fixUpperTail
                                               else '',
                                               col,
                                               _TS_WINDOW_NAMES[window]))
                        tsWindowDefs.add(_TS_WINDOW_DEFS[window])

                        if window != 'partition':
                            oppositeWindow = _TS_OPPOSITE_WINDOW_NAMES[window]
                            fallbackStrs.insert(1,
                                                partitionFallBackStrTemplate
                                                % (_TS_OPPOSITE_METHODS[methodForCol],
                                                   col,
                                                   '' if lowerNull is None
                                                   else ' OR ({} <= {})'.format(col, lowerNull),
                                                   '' if upperNull is None
                                                   else ' OR ({} >= {})'.format(col, upperNull),
                                                   ' OR (%s < %s)' % (col, valFormatter % self.outlierRstMin(col))
                                                   if fixLowerTail
                                                   else '',
                                                   ' OR (%s > %s)' % (col, valFormatter % self.outlierRstMax(col))
                                                   if fixUpperTail
                                                   else '',
                                                   col,
                                                   _TS_WINDOW_NAMES[oppositeWindow]))
                            tsWindowDefs.add(_TS_WINDOW_DEFS[oppositeWindow])

                    details[col] = \
                        [self._NULL_FILL_PREFIX + col + self._PREP_SUFFIX,

                         dict(SQL="COALESCE(CASE WHEN (STRING({0}) = 'NaN'){1}{2}{3}{4} THEN NULL ELSE {0} END, {5})"
                             .format(
                             col,
                             '' if lowerNull is None
                             else ' OR ({} <= {})'.format(col, lowerNull),
                             '' if upperNull is None
                             else ' OR ({} >= {})'.format(col, upperNull),
                             ' OR ({} < {})'.format(col, valFormatter % self.outlierRstMin(col))
                             if isNum and (col in fillOutliers) and fixLowerTail
                             else '',
                             ' OR ({} > {})'.format(col, valFormatter % self.outlierRstMax(col))
                             if isNum and (col in fillOutliers) and fixUpperTail
                             else '',
                             ', '.join(fallbackStrs)),

                             Nulls=colNulls,
                             NullFillValue=colFallBackVal)]

            if tsWindowDefs:
                details['__TS_WINDOW_CLAUSE__'] = \
                    _tsWindowClause = \
                    'WINDOW {}'.format(', '.join(tsWindowDefs))

                if self._detPrePartitioned:
                    _tsWindowClause = \
                        _tsWindowClause.replace(
                            'PARTITION BY {}, {}'.format(self._iCol, self._T_CHUNK_COL),
                            'PARTITION BY {}'.format(self._iCol))

            else:
                _tsWindowClause = ''

            sqlTransformer = \
                SQLTransformer(
                    statement=
                    'SELECT *, {} FROM __THIS__ {}'
                        .format(
                        ', '.join(
                            '{} AS {}'.format(nullFillDetails['SQL'], nullFillCol)
                            for col, (nullFillCol, nullFillDetails) in details.items()
                            if col != '__TS_WINDOW_CLAUSE__'),
                        _tsWindowClause))

        if savePath and (savePath != loadPath):
            if verbose:
                msg = 'Saving NULL-Filling SQL Transformations to HDFS Paths {}...'.format(savePath)
                self.stdout_logger.info(msg)
                _tic = time.time()

            fs.rm(
                path=savePath,
                hdfs=arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS,
                is_dir=True,
                hadoop_home=arimo.backend._HADOOP_HOME)

            sqlTransformer.save(   # *** NEED TO ENHANCE TO ALLOW OVERWRITING ***
                path=savePath)

            if verbose:
                _toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - _tic))

        adf = self(
            sqlTransformer.transform,
            inheritNRows=True,
            **kwargs)

        adf._inheritCache(
            self,
            *(() if loadPath
              else cols))

        adf._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdout_logger.info(message + ' done!   <{:,.1f} m>'.format((toc - tic) / 60))

        return ((adf, details, sqlTransformer)
                if returnSQLTransformer
                else (adf, details)) \
            if returnDetails \
            else adf

    @_docstr_verbose
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
                - **forceCat** *(str or list/tuple of str, default = None)*: columns to force to be categorical variables

                - **forceNum** *(str or list/tuple of str, default = None)*: columns to force to be numerical variables

                - **fill**:
                    - *dict* ( ``method`` = ... *(default: 'mean')*, ``value`` = ... *(default: None)*, ``outlierTails`` = ... *(default: False)*, ``fillOutliers`` = ... *(default: False)*) as per ``.fillna(...)`` method;
                    - *OR* ``None`` to not apply any ``NULL``/``NaN``-filling

                - **scaler** *(str)*: one of the following methods to use on numerical columns
                    (*ignored* if loading existing ``prep`` pipeline from ``loadPath``):

                    - ``standard`` (default)
                    - ``maxabs``
                    - ``minmax``
                    - ``None`` *(do not apply any scaling)*

                - **assembleVec** *(str, default = '__X__')*: name of vector column to build from pre-processed features; *ignored* if loading existing ``prep`` pipeline from ``loadPath``

                - **loadPath** *(str)*: HDFS path to load existing data transformations

                - **savePath** *(str)*: HDFS path to save new fitted data transformations
        """
        def sqlStdScl(sqlItem, mean, std):
            return '(({}) - {}) / {}'.format(sqlItem, mean, std)

        def sqlMaxAbsScl(sqlItem, maxAbs):
            return '({}) / {}'.format(sqlItem, maxAbs)

        def sqlMinMaxScl(sqlItem, origMin, origMax, targetMin, targetMax):
            origRange = origMax - origMin
            targetRange = targetMax - targetMin
            return '({} * (({}) - ({})) / {}) + ({})'.format(
                targetRange, sqlItem, origMin, origRange, targetMin)

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

        oheCat = kwargs.pop('oheCat', False)
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

        assert fill, '*** {}.prep(...) MUST INVOLVE NULL-FILLING FOR NUMERIC COLS ***'.format(type(self))

        scaler = kwargs.pop('scaler', 'standard')
        if scaler:
            scaler = scaler.lower()

        vecColsToAssemble = kwargs.pop('assembleVec', None)

        if not vecColsToAssemble:
            oheCat = False

        if oheCat:
            scaleCat = False

        returnOrigToPrepColMaps = kwargs.pop('returnOrigToPrepColMaps', False)
        returnPipeline = kwargs.pop('returnPipeline', False)

        loadPath = kwargs.pop('loadPath', None)
        savePath = kwargs.pop('savePath', None)

        verbose = kwargs.pop('verbose', False)
        if arimo.debug.ON:
            verbose = True

        if loadPath:
            if verbose:
                message = 'Loading & Applying Data Transformations from HDFS Path {}...'.format(loadPath)
                self.stdout_logger.info(message)
                tic = time.time()

            if loadPath in self._PREP_CACHE:
                prepCache = self._PREP_CACHE[loadPath]

                sqlTransformer = prepCache.sqlTransformer
                catOHETransformer = prepCache.catOHETransformer
                pipelineModelWithoutVectors = prepCache.pipelineModelWithoutVectors

                catOrigToPrepColMap = prepCache.catOrigToPrepColMap
                numOrigToPrepColMap = prepCache.numOrigToPrepColMap

                defaultVecCols = prepCache.defaultVecCols

            else:
                if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                    localDirExists = os.path.isdir(loadPath)

                    hdfsDirExists = \
                        arimo.backend.hdfs.test(
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

                catOrigToPrepColMap = \
                    json.load(open(os.path.join(loadPath, self._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'r'))

                numOrigToPrepColMap = \
                    json.load(open(os.path.join(loadPath, self._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'r'))

                defaultVecCols = \
                    [catOrigToPrepColMap[catCol][0]
                     for catCol in sorted(set(catOrigToPrepColMap)
                                          .difference(('__OHE__', '__SCALE__')))] + \
                    [numOrigToPrepColMap[numCol][0]
                     for numCol in sorted(set(numOrigToPrepColMap)
                                          .difference(('__TS_WINDOW_CLAUSE__', '__SCALER__')))]

                try:
                    pipelineModelWithoutVectors = sqlTransformer = SQLTransformer.load(path=loadPath)

                    catOHETransformer = None

                except:
                    pipelineModel = PipelineModel.load(path=loadPath)

                    nPipelineModelStages = len(pipelineModel.stages)

                    if nPipelineModelStages == 2:
                        sqlTransformer, secondTransformer = pipelineModel.stages

                        assert isinstance(sqlTransformer, SQLTransformer), \
                            '*** {} ***'.format(sqlTransformer)

                        if isinstance(secondTransformer, OneHotEncoderModel):
                            catOHETransformer = secondTransformer
                            vectorAssembler = None
                            pipelineModelWithoutVectors = pipelineModel

                        elif isinstance(secondTransformer, VectorAssembler):
                            catOHETransformer = None
                            vectorAssembler = secondTransformer
                            pipelineModelWithoutVectors = sqlTransformer

                        else:
                            raise ValueError('*** {} ***'.format(secondTransformer))

                    elif nPipelineModelStages == 3:
                        sqlTransformer, catOHETransformer, vectorAssembler = pipelineModel.stages

                        assert isinstance(sqlTransformer, SQLTransformer), \
                            '*** {} ***'.format(sqlTransformer)

                        assert isinstance(catOHETransformer, OneHotEncoderModel), \
                            '*** {} ***'.format(catOHETransformer)

                        assert isinstance(vectorAssembler, VectorAssembler), \
                            '*** {} ***'.format(vectorAssembler)

                        pipelineModelWithoutVectors = \
                            PipelineModel(stages=[sqlTransformer, catOHETransformer])

                    else:
                        raise ValueError('*** {} ***'.format(pipelineModel.stages))

                    if vectorAssembler:
                        vecInputCols = vectorAssembler.getInputCols()

                        assert set(defaultVecCols) == set(vecInputCols)

                        defaultVecCols = vecInputCols

                self._PREP_CACHE[loadPath] = \
                    Namespace(
                        sqlTransformer=sqlTransformer,
                        catOHETransformer=catOHETransformer,
                        pipelineModelWithoutVectors=pipelineModelWithoutVectors,

                        catOrigToPrepColMap=catOrigToPrepColMap,
                        numOrigToPrepColMap=numOrigToPrepColMap,

                        defaultVecCols=defaultVecCols)

        else:
            if cols:
                cols = set(cols)

                cols = cols.intersection(self.possibleFeatureTAuxCols).union(
                    possibleFeatureContentCol for possibleFeatureContentCol in cols.intersection(self.possibleFeatureContentCols)
                    if self.suffNonNull(possibleFeatureContentCol))

            else:
                cols = self.possibleFeatureTAuxCols + \
                       tuple(possibleFeatureContentCol for possibleFeatureContentCol in self.possibleFeatureContentCols
                             if self.suffNonNull(possibleFeatureContentCol))

            if cols:
                profile = \
                    self.profile(
                        *cols,
                        profileCat=True,
                        profileNum=False,   # or bool(fill) or bool(scaler)?
                        skipIfInvalid=True,
                        asDict=True,
                        verbose=verbose)

            else:
                return self.copy()

            cols = {col for col in cols
                    if self.suffNonNull(col) and
                    (len(profile[col].distinctProportions
                         .loc[# (profile[col].distinctProportions.index != '') &
                             # FutureWarning: elementwise comparison failed; returning scalar instead,
                             # but in the future will perform elementwise comparison
                             pandas.notnull(profile[col].distinctProportions.index)]) > 1)}

            if not cols:
                return self.copy()

            catCols = \
                [col for col in cols.intersection(self.possibleCatCols).difference(forceNum)
                 if (col in forceCat) or
                 (profile[col].distinctProportions.iloc[:self._maxNCats[col]].sum()
                  >= self._minProportionByMaxNCats[col])]

            numCols = [col for col in cols.difference(catCols)
                       if self.typeIsNum(col)]

            cols = catCols + numCols

            if verbose:
                message = 'Prepping Columns {}...'.format(', '.join('"{}"'.format(col) for col in cols))
                self.stdout_logger.info(message)
                tic = time.time()

            prepSqlItems = {}

            catOrigToPrepColMap = \
                dict(__OHE__=oheCat,
                     __SCALE__=scaleCat)

            if catCols:
                if verbose:
                    msg = 'Transforming Categorical Features {}...'.format(
                        ', '.join('"{}"'.format(catCol) for catCol in catCols))
                    self.stdout_logger.info(msg)
                    _tic = time.time()

                catIdxCols = []
                if oheCat:
                    catOHECols = []
                elif scaleCat:
                    catScaledIdxCols = []

                for catCol in catCols:
                    catIdxCol = self._CAT_IDX_PREFIX + catCol + self._PREP_SUFFIX

                    catColType = self.type(catCol)
                    isBool = (catColType == _BOOL_TYPE)

                    if isBool:
                        cats = [0, 1]

                        nCats = 2

                        catIdxSqlItem = \
                            'CASE WHEN {0} IS NULL THEN 2 \
                                  WHEN {0} THEN 1 \
                                  ELSE 0 END'.format(catCol)

                    else:
                        isStr = (catColType == _STR_TYPE)

                        cats = [cat for cat in
                                (profile[catCol].distinctProportions.index
                                 if catCol in forceCat
                                 else profile[catCol].distinctProportions.index[:self._maxNCats[catCol]])
                                if (cat != '') and pandas.notnull(cat)]

                        nCats = len(cats)

                        catIdxSqlItem = \
                            'CASE {} ELSE {} END'.format(
                                ' '.join('WHEN {} = {} THEN {}'.format(
                                    catCol,
                                    "'{}'".format(cat.replace("'", "''").replace('"', '""'))
                                    if isStr
                                    else cat,
                                    i)
                                         for i, cat in enumerate(cats)),
                                nCats)

                    if oheCat:
                        catIdxCols.append(catIdxCol)

                        prepSqlItems[catIdxCol] = catIdxSqlItem

                        catPrepCol = self._OHE_PREFIX + catCol + self._PREP_SUFFIX
                        catOHECols.append(catPrepCol)

                    elif scaleCat:
                        catPrepCol = self._MIN_MAX_SCL_PREFIX + self._CAT_IDX_PREFIX + catCol + self._PREP_SUFFIX
                        catScaledIdxCols.append(catPrepCol)

                        prepSqlItems[catPrepCol] = \
                            sqlMinMaxScl(
                                sqlItem=catIdxSqlItem,
                                origMin=0, origMax=nCats,
                                targetMin=-1, targetMax=1)

                    else:
                        prepSqlItems[catIdxCol] = catIdxSqlItem

                        catPrepCol = catIdxCol

                    catOrigToPrepColMap[catCol] = \
                        [catPrepCol,

                         dict(Cats=cats,
                              NCats=nCats)]

                if oheCat:
                    catOHETransformer = \
                        OneHotEncoder(
                            inputCols=catIdxCols,
                            outputCols=catOHECols,
                            handleInvalid='error',
                            # 'keep': invalid data presented as an extra categorical feature
                            # When handleInvalid is configured to 'keep',
                            # an extra "category" indicating invalid values is added as last category,
                            # so when dropLast is true, invalid values are encoded as all-zeros vector
                            dropLast=True) \
                            .fit(dataset=self.reprSample._sparkDF[catCols].union(
                            arimo.backend.spark.sql(
                                'VALUES ({})'.format(', '.join(len(catCols) * ('NULL',)))))
                                 .selectExpr(*('{} AS {}'.format(catSqlItem, strIdxCol)
                                               for strIdxCol, catSqlItem in prepSqlItems.items())))

                else:
                    catOHETransformer = None

                if verbose:
                    _toc = time.time()
                    self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - tic))

            else:
                catOHETransformer = None

            numOrigToPrepColMap = \
                dict(__SCALER__=scaler)

            if numCols:
                numScaledCols = []

                if verbose:
                    msg = 'Transforming Numerical Features {}...'.format(
                        ', '.join('"{}"'.format(numCol) for numCol in numCols))
                    self.stdout_logger.info(msg)
                    _tic = time.time()

                outlierTails = fill.get('outlierTails', {})
                if isinstance(outlierTails, _STR_CLASSES):
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
                        nulColNullFillValue = numColNullFillDetails['NullFillValue']

                        if scaler:
                            if scaler == 'standard':
                                scaledCol = self._STD_SCL_PREFIX + numCol + self._PREP_SUFFIX

                                mean = self.outlierRstStat(numCol)

                                stdDev = \
                                    self.reprSample \
                                        ._nonNullCol(
                                        col=numCol,
                                        lower=colMin if excludeLowerTail else None,
                                        upper=colMax if excludeUpperTail else None,
                                        strict=False) \
                                        .select(sparkSQLFuncs.stddev(numCol)) \
                                        .first()[0]

                                prepSqlItems[scaledCol] = \
                                    sqlStdScl(
                                        sqlItem=numColSqlItem,
                                        mean=mean,
                                        std=stdDev)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=nulColNullFillValue,
                                          Mean=mean,
                                          StdDev=stdDev)]

                            elif scaler == 'maxabs':
                                scaledCol = self._MAX_ABS_SCL_PREFIX + numCol + self._PREP_SUFFIX

                                maxAbs = max(abs(colMin), abs(colMax))

                                prepSqlItems[scaledCol] = \
                                    sqlMaxAbsScl(
                                        sqlItem=numColSqlItem,
                                        maxAbs=maxAbs)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=nulColNullFillValue,
                                          MaxAbs=maxAbs)]

                            elif scaler == 'minmax':
                                scaledCol = self._MIN_MAX_SCL_PREFIX + numCol + self._PREP_SUFFIX

                                prepSqlItems[scaledCol] = \
                                    sqlMinMaxScl(
                                        sqlItem=numColSqlItem,
                                        origMin=colMin, origMax=colMax,
                                        targetMin=-1, targetMax=1)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=nulColNullFillValue,
                                          OrigMin=colMin, OrigMax=colMax,
                                          TargetMin=-1, TargetMax=1)]

                            else:
                                raise ValueError('*** Scaler must be one of "standard", "maxabs", "minmax" and None ***')

                        else:
                            scaledCol = self._NULL_FILL_PREFIX + numCol + self._PREP_SUFFIX

                            prepSqlItems[scaledCol] = numColSqlItem

                            numOrigToPrepColMap[numCol] = \
                                [scaledCol,

                                 dict(Nulls=numColNulls,
                                      NullFillValue=nulColNullFillValue)]

                        numScaledCols.append(scaledCol)

                if verbose:
                    _toc = time.time()
                    self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - _tic))

            defaultVecCols = \
                [catOrigToPrepColMap[catCol][0]
                 for catCol in sorted(set(catOrigToPrepColMap)
                                      .difference(('__OHE__', '__SCALE__')))] + \
                [numOrigToPrepColMap[numCol][0]
                 for numCol in sorted(set(numOrigToPrepColMap)
                                      .difference(('__TS_WINDOW_CLAUSE__', '__SCALER__')))]

            sqlTransformer = \
                SQLTransformer(
                    statement=
                    'SELECT *, {} FROM __THIS__ {}'.format(
                        ', '.join('{} AS {}'.format(sqlItem, prepCol)
                                  for prepCol, sqlItem in prepSqlItems.items()),
                        numNullFillDetails.get('__TS_WINDOW_CLAUSE__', '')))

            pipelineModelWithoutVectors = \
                PipelineModel(stages=[sqlTransformer, catOHETransformer]) \
                    if catCols and oheCat \
                    else sqlTransformer

        if savePath and (savePath != loadPath):
            if verbose:
                msg = 'Saving Data Transformations to Local Path {}...'.format(savePath)
                self.stdout_logger.info(msg)
                _tic = time.time()

            # *** NEED TO ENHANCE TO ALLOW OVERWRITING ***
            fs.rm(
                path=savePath,
                hdfs=arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS,
                is_dir=True,
                hadoop_home=arimo.backend._HADOOP_HOME)

            pipelineModelWithoutVectors.save(path=savePath)

            if arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS:
                fs.get(
                    from_hdfs=savePath,
                    to_local=savePath,
                    is_dir=True,
                    _mv=False)

            json.dump(
                catOrigToPrepColMap,
                open(os.path.join(savePath, self._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'w'),
                indent=4)

            json.dump(
                numOrigToPrepColMap,
                open(os.path.join(savePath, self._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'w'),
                indent=4)

            if verbose:
                _toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - _tic))

            self._PREP_CACHE[savePath] = \
                Namespace(
                    sqlTransformer=sqlTransformer,
                    catOHETransformer=catOHETransformer,
                    pipelineModelWithoutVectors=pipelineModelWithoutVectors,

                    catOrigToPrepColMap=catOrigToPrepColMap,
                    numOrigToPrepColMap=numOrigToPrepColMap,

                    defaultVecCols=defaultVecCols)

        if self._detPrePartitioned and self.hasTS:
            _partitionBy_str = 'PARTITION BY {}, {}'.format(self._iCol, self._T_CHUNK_COL)

            statement = sqlTransformer.getStatement()

            if _partitionBy_str in statement:
                sqlTransformer = \
                    SQLTransformer(
                        statement=
                        statement.replace(
                            _partitionBy_str,
                            'PARTITION BY {}'.format(self._iCol)))

        pipelineModelStages = \
            [sqlTransformer] + \
            ([catOHETransformer]
             if catOHETransformer
             else [])

        if vecColsToAssemble:
            if isinstance(vecColsToAssemble, _STR_CLASSES):
                pipelineModelStages.append(
                    VectorAssembler(
                        inputCols=defaultVecCols,
                        outputCol=vecColsToAssemble))

            else:
                assert isinstance(vecColsToAssemble, (dict, Namespace))

                for vecOutputCol, vecInputCols in vecColsToAssemble.items():
                    pipelineModelStages.append(
                        VectorAssembler(
                            inputCols=vecInputCols
                            if vecInputCols
                            else defaultVecCols,
                            outputCol=vecOutputCol))

        pipelineModel = \
            PipelineModel(stages=pipelineModelStages) \
                if len(pipelineModelStages) > 1 \
                else sqlTransformer

        try:   # in case SELF is FilesBasedADF
            sparkDF = self._initSparkDF
        except:
            sparkDF = self._sparkDF

        missingCatCols = \
            set(catOrigToPrepColMap) \
                .difference(
                sparkDF.columns +
                ['__OHE__', '__SCALE__'])

        missingNumCols = \
            set(numOrigToPrepColMap) \
                .difference(
                sparkDF.columns +
                ['__TS_WINDOW_CLAUSE__', '__SCALER__'])

        if missingCatCols or missingNumCols:
            if arimo.debug.ON:
                self.stdout_logger.debug(
                    msg='*** FILLING MISSING COLS {} ***'
                        .format(missingCatCols | missingNumCols))

            sparkDF = \
                sparkDF.select(
                    '*',
                    *([sparkSQLFuncs.lit(None)
                      .alias(missingCatCol)
                       for missingCatCol in missingCatCols] +
                      [sparkSQLFuncs.lit(numOrigToPrepColMap[missingNumCol][1]['NullFillValue'])
                      .alias(missingNumCol)
                       for missingNumCol in missingNumCols]))

        colsToKeep = \
            sparkDF.columns + \
            (to_iterable(vecColsToAssemble, iterable_type=list)
             if vecColsToAssemble
             else (([catPrepColDetails[0]
                     for catCol, catPrepColDetails in catOrigToPrepColMap.items()
                     if (catCol not in ('__OHE__', '__SCALE__')) and
                     isinstance(catPrepColDetails, list) and (len(catPrepColDetails) == 2)] +
                    [numPrepColDetails[0]
                     for numCol, numPrepColDetails in numOrigToPrepColMap.items()
                     if (numCol not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and
                     isinstance(numPrepColDetails, list) and (len(numPrepColDetails) == 2)])
                   if loadPath
                   else (((catScaledIdxCols
                           if scaleCat
                           else catIdxCols)
                          if catCols
                          else []) +
                         (numScaledCols
                          if numCols
                          else []))))

        adf = self._decorate(
            obj=pipelineModel.transform(dataset=sparkDF)[colsToKeep],
            nRows=self._cache.nRows,
            **kwargs)

        adf._inheritCache(
            self,
            *(() if loadPath
              else colsToKeep))

        adf._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdout_logger.info(message + ' done!   <{:,.1f} m>'.format((toc - tic) / 60))

        return ((adf, catOrigToPrepColMap, numOrigToPrepColMap, pipelineModel)
                if returnPipeline
                else (adf, catOrigToPrepColMap, numOrigToPrepColMap)) \
            if returnOrigToPrepColMaps \
            else adf

    # *******************************
    # ITERATIVE GENERATION / SAMPLING
    # sample
    # gen

    # ****
    # MISC
    # rename

    def rename(self, **kwargs):
        """
        Return:
            ``ADF`` with new column names

        Args:
            **kwargs: arguments of the form ``newColName`` = ``existingColName``
        """
        sparkDF = self._sparkDF
        iCol = self._iCol
        tCol = self._tCol

        for newColName, existingColName in kwargs.items():
            if existingColName not in self._T_AUX_COLS:
                if existingColName == iCol:
                    iCol = newColName

                elif existingColName == tCol:
                    tCol = newColName

                sparkDF = \
                    sparkDF.withColumnRenamed(
                        existing=existingColName,
                        new=newColName)

        return self._decorate(
            obj=sparkDF, nRows=self._cache.nRows,
            iCol=iCol, tCol=tCol)


    # ********************************
    # "INTERNAL / DON'T TOUCH" METHODS
    # _inplace

    def _inplace(self, fadf, alias=None):
        if isinstance(fadf, (tuple, list)):   # just in case we're taking in multiple inputs
            fadf = fadf[0]

        assert isinstance(fadf, FileADF)

        self.path = fadf.path

        self.__dict__.update(self._CACHE[fadf.path])

        self._initSparkDF = fadf._initSparkDF
        self._sparkDFTransforms = fadf._sparkDFTransforms
        self._pandasDFTransforms = fadf._pandasDFTransforms
        self._sparkDF = fadf._sparkDF

        self.alias = alias \
            if alias \
            else (self._alias
                  if self._alias
                  else fadf._alias)

        self._cache = fadf._cache

    # **********
    # TRANSFORMS
    # transform
    # fillna
    # prep
    # drop
    # filter

    def transform(self, sparkDFTransform, _sparkDF=None, pandasDFTransform=[], *args, **kwargs):
        stdKwArgs = self._extractStdKwArgs(kwargs, resetToClassDefaults=False, inplace=False)

        if stdKwArgs.alias and (stdKwArgs.alias == self.alias):
            stdKwArgs.alias = None

        inheritCache = kwargs.pop('inheritCache', False)

        if isinstance(sparkDFTransform, list):
            additionalSparkDFTransforms = sparkDFTransform

            inheritCache |= \
                all(isinstance(additionalSparkDFTransform, Transformer)
                    for additionalSparkDFTransform in additionalSparkDFTransforms)

        elif isinstance(sparkDFTransform, Transformer):
            additionalSparkDFTransforms = [sparkDFTransform]
            inheritCache = True

        else:
            additionalSparkDFTransforms = \
                [(lambda sparkDF: sparkDFTransform(sparkDF, *args, **kwargs))
                 if args or kwargs
                 else sparkDFTransform]

        additionalPandasDFTransforms = \
            pandasDFTransform \
                if isinstance(pandasDFTransform, list) \
                else [pandasDFTransform]

        inheritNRows = kwargs.pop('inheritNRows', inheritCache)

        if _sparkDF is None:
            _sparkDF = self._sparkDF

            for i, additionalSparkDFTransform in enumerate(additionalSparkDFTransforms):
                try:
                    _sparkDF = additionalSparkDFTransform.transform(dataset=_sparkDF) \
                        if isinstance(additionalSparkDFTransform, Transformer) \
                        else additionalSparkDFTransform(_sparkDF)

                except Exception as err:
                    self.stdout_logger.error(
                        msg='*** {} ADDITIONAL TRANSFORM #{} ({}): ***'
                            .format(self.path, i, additionalSparkDFTransform))
                    raise err

        fadf = FileADF(
            path=self.path,
            _initSparkDF=self._initSparkDF,
            _sparkDFTransforms=self._sparkDFTransforms + additionalSparkDFTransforms,
            _pandasDFTransforms=self._pandasDFTransforms + additionalPandasDFTransforms,
            _sparkDF=_sparkDF,
            nRows=self._cache.nRows
            if inheritNRows
            else None,
            **stdKwArgs.__dict__)

        if inheritCache:
            fadf._inheritCache(self)

        fadf._cache.pieceADFs = self._cache.pieceADFs

        return fadf

    def drop(self, *cols, **kwargs):
        return self.transform(
            sparkDFTransform=
            lambda sparkDF:
            sparkDF.drop(*cols),
            pandasDFTransform=_FileADF__drop__pandasDFTransform(cols=cols),
            inheritCache=True,
            inheritNRows=True,
            **kwargs)

    def filter(self, condition, **kwargs):
        return self.transform(
            sparkDFTransform=
            lambda sparkDF:
            sparkDF.filter(
                condition=condition),
            pandasDFTransform=[],   # no Pandas equivalent
            inheritCache=True,
            inheritNRows=True,
            **kwargs)

    # **************
    # SUBSET METHODS
    # _subset
    # filterByPartitionKeys
    # sample
    # gen

    def _subset(self, *pieceSubPaths, **kwargs):
        if pieceSubPaths:
            assert self.pieceSubPaths.issuperset(pieceSubPaths)

            nPieceSubPaths = len(pieceSubPaths)

            if nPieceSubPaths == self.nPieces:
                return self

            else:
                verbose = kwargs.pop('verbose', True)

                if self.s3Client:
                    subsetDirS3Key = \
                        os.path.join(
                            self.tmpDirS3Key,
                            str(uuid.uuid4()))

                    subsetDirPath = \
                        os.path.join(
                            's3://{}'.format(self.s3Bucket),
                            subsetDirS3Key)

                    for pieceSubPath in \
                            (tqdm.tqdm(pieceSubPaths)
                            if verbose
                            else pieceSubPaths):
                        self.s3Client.copy(
                            CopySource=dict(
                                Bucket=self.s3Bucket,
                                Key=os.path.join(self.pathS3Key, pieceSubPath)),
                            Bucket=self.s3Bucket,
                            Key=os.path.join(subsetDirS3Key, pieceSubPath))

                    aws_access_key_id = self._srcArrowDS.fs.fs.key
                    aws_secret_access_key = self._srcArrowDS.fs.fs.secret

                else:
                    subsetDirPath = \
                        os.path.join(
                            self.tmpDirPath,
                            str(uuid.uuid4()))

                    for pieceSubPath in \
                            (tqdm.tqdm(pieceSubPaths)
                            if verbose
                            else pieceSubPaths):
                        fs.cp(
                            from_path=os.path.join(self.path, pieceSubPath),
                            to_path=os.path.join(subsetDirPath, pieceSubPath),
                            hdfs=fs._ON_LINUX_CLUSTER_WITH_HDFS, is_dir=False)

                    aws_access_key_id = aws_secret_access_key = None

                stdKwArgs = self._extractStdKwArgs(kwargs, resetToClassDefaults=False, inplace=False)

                if stdKwArgs.alias and (stdKwArgs.alias == self.alias):
                    stdKwArgs.alias = None

                if stdKwArgs.detPrePartitioned:
                    stdKwArgs.nDetPrePartitions = nPieceSubPaths

                fadf = FileADF(
                    path=subsetDirPath,
                    aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                    _srcSparkDFSchema=self._srcSparkDFSchema,
                    _sparkDFTransforms=self._sparkDFTransforms,
                    _pandasDFTransforms=self._pandasDFTransforms,
                    verbose=verbose,
                    **stdKwArgs.__dict__)

                fadf._cache.colWidth.update(self._cache.colWidth)

                return fadf

        else:
            return self

    def _pieceADF(self, pieceSubPath):
        pieceADF = self._cache.pieceADFs.get(pieceSubPath)

        if pieceADF is None:
            if self._partitionedByDateOnly:
                if self.s3Client:
                    aws_access_key_id = self._srcArrowDS.fs.fs.key
                    aws_secret_access_key = self._srcArrowDS.fs.fs.secret

                else:
                    aws_access_key_id = aws_secret_access_key = None

                stdKwArgs = self._extractStdKwArgs({}, resetToClassDefaults=False, inplace=False)

                if stdKwArgs.alias:
                    assert stdKwArgs.alias == self.alias
                    stdKwArgs.alias = None

                if stdKwArgs.detPrePartitioned:
                    stdKwArgs.nDetPrePartitions = 1

                piecePath = os.path.join(self.path, pieceSubPath)

                pieceADF = \
                    FileADF(
                        path=piecePath,
                        aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                        _srcSparkDFSchema=self._srcSparkDFSchema,
                        _sparkDFTransforms=self._sparkDFTransforms,
                        _pandasDFTransforms=self._pandasDFTransforms,
                        verbose=False,
                        **stdKwArgs.__dict__)

                pieceADF._cache.colWidth.update(self._cache.colWidth)

            else:
                pieceADF = self._subset(pieceSubPath, verbose=False)

            self._cache.pieceADFs[pieceSubPath] = pieceADF

        else:
            pieceADF._sparkDFTransforms = sparkDFTransforms = self._sparkDFTransforms

            pieceADF._sparkDF = pieceADF._initSparkDF

            for i, sparkDFTransform in enumerate(sparkDFTransforms):
                try:
                    pieceADF._sparkDF = sparkDFTransform(pieceADF._sparkDF)

                except Exception as err:
                    self.stdout_logger.error(
                        msg='*** {} TRANSFORM #{}: ***'
                            .format(pieceSubPath, i))
                    raise err

            pieceADF._pandasDFTransforms = self._pandasDFTransforms

            pieceADF._cache.type = self._cache.type

        return pieceADF

    def _pieceArrowTable(self, pieceSubPath):
        return _FileADF__pieceArrowTableFunc(
            path=self.path,
            aws_access_key_id=self._srcArrowDS.fs.fs.key,
            aws_secret_access_key=self._srcArrowDS.fs.fs.secret)(pieceSubPath)

    def _piecePandasDF(self, pieceSubPath):
        pandasDF = \
            self._pieceArrowTable(pieceSubPath) \
                .to_pandas(
                nthreads=max(1, psutil.cpu_count(logical=True) // 2),
                strings_to_categorical=False,
                memory_pool=None,
                zero_copy_only=True)

        if self._tCol:
            pandasDF = \
                gen_aux_cols(
                    df=pandasDF,
                    i_col=self._iCol,
                    t_col=self._tCol)

        for i, pandasDFTransform in enumerate(self._pandasDFTransforms):
            try:
                pandasDF = pandasDFTransform(pandasDF)

            except Exception as err:
                self.stdout_logger.error(
                    msg='*** "{}": PANDAS TRANSFORM #{} ***'
                        .format(pieceSubPath, i))
                raise err

        return pandasDF

    def filterByPartitionKeys(self, *filterCriteriaTuples, **kwargs):
        filterCriteria = {}
        _samplePieceSubPath = next(iter(self.pieceSubPaths))

        for filterCriteriaTuple in filterCriteriaTuples:
            assert isinstance(filterCriteriaTuple, (list, tuple))
            filterCriteriaTupleLen = len(filterCriteriaTuple)

            col = filterCriteriaTuple[0]

            if '{}='.format(col) in _samplePieceSubPath:
                if filterCriteriaTupleLen == 2:
                    fromVal = toVal = None
                    inSet = {str(v) for v in to_iterable(filterCriteriaTuple[1])}

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
                        '*** {} FILTER CRITERIA MUST BE EITHER (<colName>, <fromVal>, <toVal>) OR (<colName>, <inValsSet>) ***'
                            .format(type(self)))

                filterCriteria[col] = fromVal, toVal, inSet

        if filterCriteria:
            pieceSubPaths = set()

            for pieceSubPath in self.pieceSubPaths:
                chk = True

                for col, (fromVal, toVal, inSet) in filterCriteria.items():
                    v = re.search('{}=(.*?)/'.format(col), pieceSubPath).group(1)

                    if ((fromVal is not None) and (v < fromVal)) or \
                            ((toVal is not None) and (v > toVal)) or \
                            ((inSet is not None) and (v not in inSet)):
                        chk = False
                        break

                if chk:
                    pieceSubPaths.add(pieceSubPath)

            assert pieceSubPaths, \
                '*** {}: NO PIECE PATHS SATISFYING FILTER CRITERIA {} ***'.format(self, filterCriteria)

            if arimo.debug.ON:
                self.stdout_logger.debug(
                    msg='*** {} PIECES SATISFYING FILTERING CRITERIA: {} ***'
                        .format(len(pieceSubPaths), filterCriteria))

            return self._subset(*pieceSubPaths, **kwargs)

        else:
            return self

    def gen(self, *args, **kwargs):
        return _FileADF__gen(
            args=args,
            path=self.path,
            pieceSubPaths=kwargs.get('pieceSubPaths', self.pieceSubPaths),
            aws_access_key_id=self._srcArrowDS.fs.fs.key, aws_secret_access_key=self._srcArrowDS.fs.fs.secret,
            iCol=self._iCol, tCol=self._tCol,
            possibleFeatureTAuxCols=self.possibleFeatureTAuxCols,
            contentCols=self.contentCols,
            pandasDFTransforms=self._pandasDFTransforms,
            filterConditions=kwargs.get('filter', {}),
            n=kwargs.get('n', 512),
            nSamplesPerPiece=kwargs.get('nSamplesPerPiece', 10 ** 5),
            anon=kwargs.get('anon', True),
            n_threads=kwargs.get('n_threads', 1))

    # ****
    # MISC
    # split

    def split(self, *weights):
        if (not weights) or weights == (1,):
            return self

        else:
            nWeights = len(weights)
            cumuWeights = numpy.cumsum(weights) / sum(weights)

            nPieces = self.nPieces

            pieceSubPaths = list(self.pieceSubPaths)
            random.shuffle(pieceSubPaths)

            cumuIndices = \
                [0] + \
                [int(round(cumuWeights[i] * nPieces))
                 for i in range(nWeights)]

            return [self._subset(*pieceSubPaths[cumuIndices[i]:cumuIndices[i + 1]])
                    for i in range(nWeights)]
