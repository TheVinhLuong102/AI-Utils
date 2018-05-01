from __future__ import division, print_function

import abc
import datetime
import math
import multiprocessing
import os
import pandas
import psutil
import random
import re
import time
import tqdm

import six
if six.PY2:
    from urlparse import urlparse
    _STR_CLASSES = str, unicode
else:
    from urllib.parse import urlparse
    _STR_CLASSES = str

from pyarrow.filesystem import LocalFileSystem
from pyarrow.hdfs import HadoopFileSystem
from pyarrow.parquet import ParquetDataset, read_metadata, read_schema, read_table
from s3fs import S3FileSystem

from arimo.df import _DF_ABC
from arimo.util import fs, Namespace
from arimo.util.aws import s3
from arimo.util.date_time import gen_aux_cols, DATE_COL
from arimo.util.decor import enable_inplace
import arimo.debug


class _FileDFABC(_DF_ABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_REPR_SAMPLE_N_PIECES = 100

    # file systems
    _LOCAL_ARROW_FS = LocalFileSystem()

    _HDFS_ARROW_FS = \
        HadoopFileSystem() \
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


@enable_inplace
class FileDF(_FileDFABC):
    _inplace_able = ()

    _CACHE = {}
    _PIECE_CACHES = {}

    def __init__(
            self, path=None, reCache=False,
            aws_access_key_id=None, aws_secret_access_key=None,
            iCol=None, tCol=None, defaultMapper=None,
            reprSampleNPieces=_FileDFABC._DEFAULT_REPR_SAMPLE_N_PIECES,
            reprSampleSize=_FileDFABC._DEFAULT_REPR_SAMPLE_SIZE,
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
                _cache.piecePaths = set()

                for i, piece in enumerate(_cache._arrowDS.pieces):
                    piecePath = piece.path
                    _cache.piecePaths.add(piecePath)

                    self._PIECE_CACHES[piecePath] = \
                        Namespace(
                            localOrHDFSPath=None
                                if path.startswith('s3')
                                else piecePath,
                            columns=(),
                            types=Namespace(
                                arrow=Namespace(),
                                pandas=Namespace()),
                            nRows=None)

            else:
                _cache.nPieces = 1
                _cache.piecePaths = {path}

                self._PIECE_CACHES[path] = \
                    Namespace(
                        localOrHDFSPath=None
                            if path.startswith('s3')
                            else path,
                        columns=(),
                        types=Namespace(
                            arrow=Namespace(),
                            pandas=Namespace()),
                        nRows=None)

            _cache.columns = set()
            
            _cache.types = \
                Namespace(
                    arrow=Namespace(),
                    pandas=Namespace())

            _cache._nRows = _cache._approxNRows = None

            if path.startswith('s3'):
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

        self.__dict__.update(_cache)

        self._iCol = iCol
        self._tCol = tCol
        self.hasTS = iCol and tCol

        self._defaultMapper = defaultMapper

        self._reprSampleNPieces = min(reprSampleNPieces, self.nPieces)
        self._reprSampleSize = reprSampleSize

        self._cache = \
            Namespace(
                reprSamplePiecePaths=None,
                reprSample=None)

    @property
    def _pathsRepr(self):
        return '"{}"'.format(self.path) \
            if isinstance(self.path, _STR_CLASSES) \
          else '{} Paths e.g. {}'.format(len(self.path), self.path[:3])

    def __repr__(self):
        return '{:,}-piece {} [{}]'.format(
            self.nPieces,
            type(self).__name__,
            self._pathsRepr)

    def __str__(self):
        return repr(self)

    @property
    def __short_repr__(self):
        return '{:,}-piece {} [{}]'.format(
            self.nPieces,
            type(self).__name__,
            self._pathsRepr)

    def pieceLocalOrHDFSPath(self, piecePath):
        if self._PIECE_CACHES[piecePath].localOrHDFSPath is None:
            parsedURL = \
                urlparse(
                    url=piecePath,
                    scheme='',
                    allow_fragments=True)

            localCachePath = \
                os.path.join(
                    self._TMP_DIR_PATH,
                    parsedURL.netloc,
                    parsedURL.path[1:])

            fs.mkdir(
                dir=os.path.dirname(localCachePath),
                hdfs=False)

            self.s3Client.download_file(
                Bucket=parsedURL.netloc,
                Key=parsedURL.path[1:],
                Filename=localCachePath)

            self._PIECE_CACHES[piecePath].localOrHDFSPath = localCachePath

        return self._PIECE_CACHES[piecePath].localOrHDFSPath

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
                    pieceCache.types.arrow[col] = _arrowType = \
                        pieceArrowTable.schema.field_by_name(col).type

                    if col in self.types.arrow:
                        assert _arrowType == self.types.arrow[col], \
                            '*** {} COLUMN {}: DETECTED TYPE {} != {} ***'.format(
                                piecePath, col, _arrowType, self.types.arrow[col])

                    else:
                        self.types.arrow[col] = _arrowType

                        self.types.pandas[col] = \
                            pieceCache.types.pandas[col] = \
                            _arrowType.to_pandas_dtype()

                pieceCache.nRows = pieceArrowTable.num_rows

            assert pieceCache.nRows

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

    @property
    def nRows(self):
        if self._nRows is None:
            self._nRows = \
                sum(read_metadata(where=self.pieceLocalOrHDFSPath(piecePath=piecePath)).num_rows
                    for piecePath in tqdm.tqdm(self.piecePaths))

        return self._nRows

    def collect(self, *cols, **kwargs):
        return self._mr(cols=cols if cols else None, **kwargs)

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

    @property
    def reprSamplePiecePaths(self):
        if self._cache.reprSamplePiecePaths is None:
            self._cache.reprSamplePiecePaths = \
                random.sample(
                    population=self.piecePaths,
                    k=self._reprSampleNPieces)

        return self._cache.reprSamplePiecePaths

    @property
    def approxNRows(self):
        if self._approxNRows is None:
            self._approxNRows = \
                self.nPieces \
                * sum(read_metadata(where=self.pieceLocalOrHDFSPath(piecePath=piecePath)).num_rows
                      for piecePath in tqdm.tqdm(self.reprSamplePiecePaths)) \
                / self._reprSampleNPieces

        return self._approxNRows

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

    @property
    def reprSample(self):
        if self._cache.reprSample is None:
            self._cache.reprSample = \
                self.sample(
                    n=self._reprSampleSize,
                    piecePaths=self.reprSamplePiecePaths,
                    verbose=True)

        return self._cache.reprSample

    def gen(self, *cols, **kwargs):
        pass
