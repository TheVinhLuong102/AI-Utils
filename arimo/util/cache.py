from __future__ import print_function

import abc
import joblib
import json
import os
import six

from . import fs, pkl
from .aws import s3


_STR_CLASSES = \
    (str, unicode) \
    if six.PY2 \
    else str


class _CacheDecorABC(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def pre_condition_lambda(self):
        raise NotImplementedError

    @abc.abstractproperty
    def validation_lambda(self):
        raise NotImplementedError

    @abc.abstractproperty
    def verbose(self):
        raise NotImplementedError


class SparkXDFonS3CacheDecor(_CacheDecorABC):
    def __init__(
            self,
            s3_bucket,
            s3_cache_dir_prefix,
            aws_access_key_id,
            aws_secret_access_key,
            name_lambda,
            format='parquet',
            pre_condition_lambda=None,
            validation_lambda=None,
            post_process_lambda=None,
            verbose=False):
        self.s3_bucket = s3_bucket

        self.s3_cache_dir_prefix = s3_cache_dir_prefix

        self.s3_cache_dir_path = \
            os.path.join(
                's3://{}'.format(s3_bucket),
                s3_cache_dir_prefix)

        self.aws_access_key_id = aws_access_key_id

        self.aws_secret_access_key = aws_secret_access_key

        self.s3_client = \
            s3.client(
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key)

        self.format = format

        self.name_lambda = name_lambda

        self._pre_condition_lambda = pre_condition_lambda

        self._validation_lambda = validation_lambda

        self.post_process_lambda = post_process_lambda

        self._verbose = verbose

    @property
    def pre_condition_lambda(self):
        return self._pre_condition_lambda

    @property
    def validation_lambda(self):
        return self._validation_lambda

    @property
    def verbose(self):
        return self._verbose

    def __call__(self, func):
        def decor_func(*args, **kwargs):
            _force_compute = kwargs.get('_force_compute', False)
            verbose = kwargs.get('_cache_verbose', self.verbose)

            name = self.name_lambda(*args, **kwargs)

            s3_key = \
                os.path.join(
                    self.s3_cache_dir_prefix,
                    name)

            s3_path = \
                os.path.join(
                    self.s3_cache_dir_path,
                    name)

            if not _force_compute:
                _force_compute = \
                    'Contents' not in \
                        self.s3_client.list_objects(
                            Bucket=self.s3_bucket,
                            Prefix=s3_key)

            if not _force_compute:
                if verbose:
                    print('Reading Cached Data from {}... '.format(s3_path), end='')

                from h1st.df.spark import SparkXDF

                result = SparkXDF.load(
                    path=s3_path,
                    format=self.format,
                    schema=None,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    verbose=verbose)

                if self.validation_lambda:
                    if self.validation_lambda(result):
                        if verbose:
                            print('done!')

                    else:
                        _force_compute = True

                        if verbose:
                            print('INVALID CACHED DATA TO BE RE-COMPUTED!')

            if _force_compute:
                result = func(*args, **kwargs)

                if self.validation_lambda:
                    assert self.validation_lambda(result)

                if (self.pre_condition_lambda is None) or self.pre_condition_lambda(*args, **kwargs):
                    result.save(
                        path=s3_path,
                        format=self.format,
                        aws_access_key_id=self.aws_access_key_id,
                        aws_secret_access_key=self.aws_secret_access_key,
                        verbose=verbose)

            return self.post_process_lambda(result, **kwargs) \
                if self.post_process_lambda \
                else result

        return decor_func


class S3CacheDecor(_CacheDecorABC):
    def __init__(
            self,
            s3_client,
            s3_bucket,
            s3_cache_dir_prefix,
            local_cache_dir_path,
            file_name_lambda,
            serializer='joblib',
            pre_condition_lambda=None,
            validation_lambda=None,
            post_process_lambda=None,
            verbose=False):
        self.s3_client = s3_client

        self.s3_bucket = s3_bucket

        self.s3_cache_dir_prefix = s3_cache_dir_prefix

        self.local_cache_dir_path = local_cache_dir_path
        fs.mkdir(
            dir=local_cache_dir_path,
            hdfs=False)

        self.file_name_lambda = file_name_lambda

        if isinstance(serializer, _STR_CLASSES):
            if serializer == 'json':
                self.file_read_lambda = \
                    lambda file_name: \
                        json.load(
                            open(os.path.join(local_cache_dir_path, file_name), 'r'),
                            encoding='utf-8')

                self.file_write_lambda = \
                    lambda obj, file_name: \
                        json.dump(
                            obj=obj,
                            fp=open(os.path.join(local_cache_dir_path, file_name), 'w'),
                            ensure_ascii=False,
                            allow_nan=True,
                            indent=4,
                            encoding='utf-8')

            elif serializer in ('joblib', 'pickle'):
                self.file_read_lambda = \
                    lambda file_name: \
                        joblib.load(filename=os.path.join(local_cache_dir_path, file_name))

                self.file_write_lambda = \
                    lambda obj, file_name: \
                        joblib.dump(
                            obj,
                            filename=os.path.join(local_cache_dir_path, file_name),
                            compress=(pkl.COMPAT_COMPRESS, pkl.MAX_COMPRESS_LVL),
                            protocol=pkl.COMPAT_PROTOCOL)

        else:
            self.file_read_lambda = \
                lambda file_name: \
                    serializer.load(
                        os.path.join(local_cache_dir_path, file_name))

            self.file_write_lambda = \
                lambda obj, file_name: \
                    serializer.dump(
                        obj,
                        os.path.join(local_cache_dir_path, file_name))

        self._pre_condition_lambda = pre_condition_lambda

        self._validation_lambda = validation_lambda

        self.post_process_lambda = post_process_lambda

        self._verbose = verbose

    @property
    def pre_condition_lambda(self):
        return self._pre_condition_lambda

    @property
    def validation_lambda(self):
        return self._validation_lambda

    @property
    def verbose(self):
        return self._verbose

    def __call__(self, func):
        def decor_func(*args, **kwargs):
            _force_compute = kwargs.get('_force_compute', False)
            verbose = kwargs.get('_cache_verbose', self.verbose)

            file_name = self.file_name_lambda(*args, **kwargs)

            local_cache_file_path = \
                os.path.join(
                    self.local_cache_dir_path,
                    file_name)

            s3_file_key = \
                os.path.join(
                    self.s3_cache_dir_prefix,
                    file_name)

            if not _force_compute:
                if os.path.isfile(local_cache_file_path):
                    _local_cache_file_exists = True

                else:
                    _local_cache_file_exists = False

                    if 'Contents' in \
                            self.s3_client.list_objects(
                                Bucket=self.s3_bucket,
                                Prefix=s3_file_key):
                        self.s3_client.download_file(
                            Bucket=self.s3_bucket,
                            Key=s3_file_key,
                            Filename=local_cache_file_path)

                    else:
                        _force_compute = True

            if not _force_compute:
                if verbose:
                    print('Reading cached result from {0}...'.format(local_cache_file_path), end=' ')

                result = \
                    self.file_read_lambda(
                        file_name=file_name)

                if self.validation_lambda:
                    if self.validation_lambda(result):
                        if verbose:
                            print('done!')

                    else:
                        if _local_cache_file_exists and \
                                ('Contents' in
                                    self.s3_client.list_objects(
                                        Bucket=self.s3_bucket,
                                        Prefix=s3_file_key)):
                            self.s3_client.download_file(
                                Bucket=self.s3_bucket,
                                Key=s3_file_key,
                                Filename=local_cache_file_path)

                            result = \
                                self.file_read_lambda(
                                    file_name=file_name)

                            if self.validation_lambda(result):
                                if verbose:
                                    print('done!')

                            else:
                                _force_compute = True

                                if verbose:
                                    print('INVALID CACHED RESULT TO BE RE-COMPUTED!')

                        else:
                            _force_compute = True

                            if verbose:
                                print('INVALID CACHED RESULT TO BE RE-COMPUTED!')
                    
            if _force_compute:
                result = func(*args, **kwargs)

                if self.validation_lambda:
                    assert self.validation_lambda(result)

                if (self.pre_condition_lambda is None) or self.pre_condition_lambda(*args, **kwargs):
                    self.file_write_lambda(
                        obj=result,
                        file_name=file_name)

                    self.s3_client.upload_file(
                        Filename=local_cache_file_path,
                        Bucket=self.s3_bucket,
                        Key=s3_file_key)

            return self.post_process_lambda(result, **kwargs) \
                if self.post_process_lambda \
                else result

        return decor_func


class ElasticacheCacheDecor(_CacheDecorABC):
    pass


class MemCachedCacheDecor(_CacheDecorABC):
    pass


class RedisCacheDecor(_CacheDecorABC):
    pass
