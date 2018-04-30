from __future__ import division, print_function

import abc
import datetime
import io
import multiprocessing
import os
import pandas
import psutil
import random
import re
import threading
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
from pyarrow.parquet import ParquetDataset, read_metadata, read_pandas, read_schema, read_table
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

                _pathPlusSepLen = len(path) + 1

                _cache.pieceSubPaths = set()

                for i, piece in enumerate(_cache._arrowDS.pieces):
                    piecePath = piece.path
                    _cache.piecePaths.add(piecePath)

                    pieceSubPath = piecePath[_pathPlusSepLen:]
                    _cache.pieceSubPaths.add(pieceSubPath)

                    if not i:
                        _cache._partitionedByDateOnly = \
                            pieceSubPath.startswith('{}='.format(DATE_COL)) and \
                            (pieceSubPath.count('/') == 1)

                    self._PIECE_CACHES[piecePath] = \
                        Namespace(
                            localPath=None
                                if path.startswith('s3')
                                else piecePath,
                            columns=None, types=None,
                            nRows=None)

            else:
                _cache.nPieces = 1
                _cache.piecePaths = {path}
                _cache.pieceSubPaths = {}
                _cache._partitionedByDateOnly = False

                self._PIECE_CACHES[path] = \
                    Namespace(
                        localPath=None
                            if path.startswith('s3')
                            else path,
                        columns=None, types=None,
                        nRows=None)

            _cache._cache = \
                Namespace(
                    columns=set(), types=Namespace(),
                    nRows=None, approxNRows=None)

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
        self._reprSamplePiecePaths = None
        self._reprSampleSize = reprSampleSize

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

    def pieceLocalCachePath(self, piecePath):
        if self._PIECE_CACHES[piecePath].localPath is None:
            parsed_url = \
                urlparse(
                    url=piecePath,
                    scheme='',
                    allow_fragments=True)

            localCachePath = \
                os.path.join(
                    self._TMP_DIR_PATH,
                    parsed_url.netloc,
                    parsed_url.path[1:])

            fs.mkdir(
                dir=os.path.dirname(localCachePath),
                hdfs=False)

            self.s3Client.download_file(
                Bucket=parsed_url.netloc,
                Key=parsed_url.path[1:],
                File=localCachePath)

            self._PIECE_CACHES[piecePath].localPath = localCachePath

        return self._PIECE_CACHES[piecePath].localPath

    def _mr(self, *piecePaths, **kwargs):
        cols = kwargs.get('cols')
        if not cols:
            cols = None

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
        
        if not piecePaths:
            piecePaths = self.piecePaths

        results = []

        for piecePath in tqdm.tqdm(piecePaths):
            df = pandas.read_parquet(
                    path=self.pieceLocalCachePath(piecePath=piecePath),
                    engine='pyarrow',
                    columns=cols,
                    nthreads=psutil.cpu_count(logical=True))

            for partitionKV in re.findall('[^/]+=[^/]+/', piecePath):
                k, v = partitionKV.split('=')

                df[str(k)] = datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date() \
                    if k == DATE_COL \
                    else v[:-1]

            pieceCache = self._PIECE_CACHES[piecePath]

            if pieceCache.columns is None:
                _columns = df.columns
                pieceCache.columns = _columns
                self._cache.columns.update(_columns)

            if pieceCache.types is None:
                _piece_types = df.dtypes
                pieceCache.types = _piece_types
                for col in df.columns:
                    _type = _piece_types[col]
                    if col in self._cache.types:
                        self._cache.types[col].add(_type)
                    else:
                        self._cache.types[col] = {_type}

            if pieceCache.nRows is None:
                pieceCache.nRows = len(df)
    
            if organizeTS and self._tCol:
                assert self._tCol in df.columns, \
                    '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self.tCol, df.columns)
    
                if self._iCol:
                    assert self._iCol in df.columns, \
                        '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piecePath, self.iCol, df.columns)
    
                    df = gen_aux_cols(
                        df=df.loc[pandas.notnull(df[self._iCol]) &
                                  pandas.notnull(df[self._tCol])],
                        i_col=self._iCol, t_col=self._tCol)
    
                else:
                    df = gen_aux_cols(
                        df=df.loc[pandas.notnull(df[self._tCol])],
                        i_col=None, t_col=self._tCol)

            if applyDefaultMapper and self._defaultMapper:
                df = self._defaultMapper(df)

            results.append(
                mapper(df)
                if mapper
                else df)

        return reducer(results)

    @property
    def nRows(self):
        if self._cache.nRows is None:
            self._cache.nRows = \
                sum(read_metadata(where=self.pieceLocalCachePath(piecePath=piecePath)).num_rows
                    for piecePath in tqdm.tqdm(self.piecePaths))

        return self._cache.nRows

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
        if self._reprSamplePiecePaths is None:
            self._reprSamplePiecePaths = \
                random.sample(
                    population=self.piecePaths,
                    k=self._reprSampleNPieces)

        return self._reprSamplePiecePaths

    @property
    def approxNRows(self):
        if self._cache.approxNRows is None:
            self._cache.approxNRows = \
                sum(read_metadata(where=self.pieceLocalCachePath(piecePath=piecePath)).num_rows
                    for piecePath in tqdm.tqdm(self.reprSamplePiecePaths))

        return self._cache.approxNRows

    @property
    def reprSample(self):
        if self._cache.reprSample is None:
            i = 0
            _dfs = []
            n_samples = 0

            while n_samples < self.reprSampleSize:
                _repr_sample_file_path = self.reprSamplePiecePaths[i]

                _next_i = i + 1

                msg = 'Sampling from File #{:,}/{:,}: "{}"...'.format(
                    _next_i, self.repr_sample_n_files, _repr_sample_file_path)

                print(msg)



                _n_samples = min(self.reprSampleSize // self.repr_sample_n_files, len(_df))

                _df = _df.sample(
                    n=_n_samples,
                    replace=False,
                    weights=None,
                    random_state=None,
                    axis='index')

                _dfs.append(_df)

                n_samples += _n_samples

                print(msg + ' {:,} samples'.format(_n_samples))

                i = _next_i

            self._cache.reprSample = \
                pandas.concat(
                    objs=_dfs,
                    axis='index',
                    join='outer',
                    join_axes=None,
                    ignore_index=True,
                    keys=None,
                    levels=None,
                    names=None,
                    verify_integrity=False,
                    copy=False)

        return self._cache.reprSample

    def gen(self, *cols, **kwargs):
        pass

    def sample(self, *cols, **kwargs):
        n = kwargs.pop('n', 1)

        file_paths

        return pandas.concat(
            objs=_dfs,
            axis='index',
            join='outer',
            join_axes=None,
            ignore_index=True,
            keys=None,
            levels=None,
            names=None,
            verify_integrity=False,
            copy=False)
