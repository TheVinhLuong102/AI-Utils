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
from pyarrow.parquet import ParquetDataset
from s3fs import S3FileSystem

from arimo.df import _DF_ABC
from arimo.util import Namespace
from arimo.util.aws import s3
from arimo.util.date_time import gen_aux_cols, DATE_COL
from arimo.util.decor import enable_inplace
import arimo.debug


class _FileDFABC(_DF_ABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_REPR_SAMPLE_N_PIECES = 100


@enable_inplace
class FileDF(_FileDFABC):
    _inplace_able = ()

    def __init__(
            self, paths=None, _arrow_ds=None, local_cache_dir_path=_FileDFABC._TMP_DIR_PATH, _cache=None, _piece_caches=None,
            aws_access_key_id=None, aws_secret_access_key=None,
            i_col=None, t_col=None, default_map_func=None,
            repr_sample_n_pieces=_FileDFABC._DEFAULT_REPR_SAMPLE_N_PIECES,
            repr_sample_size=_FileDFABC._DEFAULT_REPR_SAMPLE_SIZE,
            verbose=True):
        if paths is None:
            assert isinstance(_arrow_ds, ParquetDataset)

            self.paths = _arrow_ds.paths

        else:
            if verbose or arimo.debug.ON:
                logger = self.class_stdout_logger()

            assert _arrow_ds is None

            self.paths = paths

            if verbose:
                msg = 'Loading Arrow Dataset from {}...'.format(
                    '"{}"'.format(self.paths)
                    if isinstance(self.paths, _STR_CLASSES)
                    else '{} Paths e.g. {}'.format(len(self.paths), paths[:3]))
                
                logger.info(msg)
                tic = time.time()

            _arrow_ds = \
                ParquetDataset(
                    path_or_paths=paths,
                    filesystem=S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key),
                    schema=None, validate_schema=False, metadata=None,
                    split_row_groups=False)

            if verbose:
                toc = time.time()
                logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

        self._arrow_ds = _arrow_ds

        self.local_cache_dir_path = local_cache_dir_path

        piece_paths = {piece.path for piece in _arrow_ds.pieces}

        if not piece_paths:
            piece_paths = {self.paths}

        self.piece_paths = piece_paths

        self.n_pieces = n_pieces = len(piece_paths)

        self._cache = \
            Namespace(
                columns=set(),
                types=Namespace(),
                n_rows=None,
                repr_sample_piece_paths=[],
                repr_sample=None)

        if _cache:
            self._cache.update(_cache)

        if _piece_caches:
            self._piece_caches = _piece_caches

            for piece_path in piece_paths.difference(_piece_caches):
                self._piece_caches[piece_path] = \
                    Namespace(
                        columns=None,
                        types=None,
                        n_rows=None)

        else:
            self._piece_caches = \
                {piece_path:
                    Namespace(
                        columns=None,
                        types=None,
                        n_rows=None)
                 for piece_path in piece_paths}

        self._i_col = i_col
        self._t_col = t_col

        self.has_ts = i_col and t_col

        self._default_map_func = default_map_func

        self._repr_sample_n_pieces = min(repr_sample_n_pieces, n_pieces)
        self._repr_sample_size = repr_sample_size

        self.s3_client = \
            s3.client(
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key)

    @property
    def _paths_repr(self):
        return '"{}"'.format(self.paths) \
            if isinstance(self.paths, _STR_CLASSES) \
          else '{} Paths e.g. {}'.format(len(self.paths), self.paths[:3])

    def __repr__(self):
        return '{:,}-piece {} [{}]'.format(
            self.n_pieces,
            type(self).__name__,
            self._paths_repr)

    def __str__(self):
        return repr(self)

    @property
    def __short_repr__(self):
        return '{:,}-piece {} [{}]'.format(
            self.n_pieces,
            type(self).__name__,
            self._paths_repr)

    @property
    def i_col(self):
        return self._i_col

    @i_col.setter
    def i_col(self, i_col):
        if i_col != self._i_col:
            self._i_col = i_col

            if i_col is None:
                self.has_ts = False
            else:
                assert i_col
                self.has_ts = bool(self._t_col)

    @i_col.deleter
    def i_col(self):
        self._i_col = None
        self.has_ts = False

    @property
    def t_col(self):
        return self._t_col

    @t_col.setter
    def t_col(self, t_col):
        if t_col != self._t_col:
            self._t_col = t_col

            if t_col is None:
                self.has_ts = False
            else:
                assert t_col
                self.has_ts = bool(self._i_col)

    @t_col.deleter
    def t_col(self):
        self._t_col = None
        self.has_ts = False

    @property
    def default_map_func(self):
        return self._default_map_func

    @default_map_func.setter
    def default_map_func(self, default_map_func):
        self._default_map_func = default_map_func

    @default_map_func.deleter
    def default_map_func(self):
        self._default_map_func = None

    def _map_reduce(self, *piece_paths, **kwargs):
        cols = kwargs.get('cols')
        if not cols:
            cols = None

        organize_ts = kwargs.get('organize_ts', True)

        apply_default_map_func = kwargs.get('apply_default_map_func', True)

        map_func = kwargs.get('map_func')

        reduce_func = \
            kwargs.get(
                'reduce_func',
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
        
        if not piece_paths:
            piece_paths = self.piece_paths

        results = []

        for piece_path in tqdm.tqdm(piece_paths):
            parsed_url = \
                urlparse(
                    url=piece_path,
                    scheme='',
                    allow_fragments=True)

            buffer = io.BytesIO()

            self.s3_client.download_fileobj(
                Bucket=parsed_url.netloc,
                Key=parsed_url.path[1:],
                Fileobj=buffer)

            df = pandas.read_parquet(
                path=buffer,
                engine='pyarrow',
                columns=cols,
                nthreads=psutil.cpu_count(logical=True))

            for partition_key_and_value in re.findall('[^/]+=[^/]+/', piece_path):
                k, v = partition_key_and_value.split('=')

                df[str(k)] = datetime.datetime.strptime(v[:-1], '%Y-%m-%d').date() \
                    if k == DATE_COL \
                    else v[:-1]

            piece_cache = self._piece_caches[piece_path]

            if piece_cache.columns is None:
                _columns = df.columns
                piece_cache.columns = _columns
                self._cache.columns.update(_columns)

            if piece_cache.types is None:
                _piece_types = df.dtypes
                piece_cache.types = _piece_types
                for col in df.columns:
                    _type = _piece_types[col]
                    if col in self._cache.types:
                        self._cache.types[col].add(_type)
                    else:
                        self._cache.types[col] = {_type}

            if piece_cache.n_rows is None:
                piece_cache.n_rows = len(df)
    
            if organize_ts and self._t_col:
                assert self._t_col in df.columns, \
                    '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piece_path, self.t_col, df.columns)
    
                if self._i_col:
                    assert self._i_col in df.columns, \
                        '*** {} DOES NOT HAVE COLUMN {} AMONG {} ***'.format(piece_path, self.i_col, df.columns)
    
                    df = gen_aux_cols(
                        df=df.loc[pandas.notnull(df[self._i_col]) &
                                  pandas.notnull(df[self._t_col])],
                        i_col=self._i_col, t_col=self._t_col)
    
                else:
                    df = gen_aux_cols(
                        df=df.loc[pandas.notnull(df[self._t_col])],
                        i_col=None, t_col=self._t_col)

            if apply_default_map_func and self._default_map_func:
                df = self._default_map_func(df)

            results.append(
                map_func(df)
                if map_func
                else df)

        return reduce_func(results)

    @property
    def n_rows(self):
        if self._cache.n_rows is None:
            self._cache.n_rows = \
                self._map_reduce(
                    cols=((self._i_col,)
                          if self._i_col
                          else ((self._t_col,)
                                if self._t_col
                                else None)),
                    organize_ts=False,
                    apply_default_map_func=False,
                    map_func=len, reduce_func=sum)

        return self._cache.n_rows

    def collect(self, *cols, **kwargs):
        return self._map_reduce(cols=cols if cols else None, **kwargs)

    @property
    def repr_sample_file_paths(self):
        if not self._repr_sample_file_paths:
            self._repr_sample_file_paths = \
                random.sample(
                    population=self.file_paths,
                    k=self.repr_sample_n_files)

        return self._repr_sample_file_paths

    @property
    def repr_sample_df(self):
        if self._repr_sample_df is None:
            i = 0
            _dfs = []
            n_samples = 0

            while n_samples < self.repr_sample_size:
                _repr_sample_file_path = self.repr_sample_file_paths[i]

                _next_i = i + 1

                msg = 'Sampling from File #{:,}/{:,}: "{}"...'.format(
                    _next_i, self.repr_sample_n_files, _repr_sample_file_path)

                print(msg)



                _n_samples = min(self.repr_sample_size // self.repr_sample_n_files, len(_df))

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

            self._repr_sample_df = \
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

        return self._repr_sample_df

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
