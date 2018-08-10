# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, unicode_literals

import abc
import argparse
import copy
import itertools
import joblib
import logging
import math
import numpy
import os
import pandas
from sklearn.preprocessing import LabelEncoder
import tempfile
import time
import tqdm
import uuid

import six
_STR_CLASSES = \
    (str, unicode) \
    if six.PY2 \
    else str

from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
import pyspark.sql
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

import arimo.backend
from arimo.df.from_files import ArrowADF, \
    _ArrowADF__castType__pandasDFTransform, _ArrowADF__encodeStr__pandasDFTransform
from arimo.df.spark import SparkADF
from arimo.df.spark_from_files import ArrowSparkADF
from arimo.dl.base import DataFramePreprocessor, ModelServingPersistence
import arimo.eval.metrics
from arimo.util import clean_str, clean_uuid, date_time, fs, import_obj, Namespace
from arimo.util.aws import s3
from arimo.util.date_time import DATE_COL, MONTH_COL, \
    _PRED_VARS_INCL_T_AUX_COLS, _PRED_VARS_INCL_T_CAT_AUX_COLS, _PRED_VARS_INCL_T_NUM_AUX_COLS
from arimo.util.iterables import to_iterable
from arimo.util.log import STDOUT_HANDLER
from arimo.util.pkl import COMPAT_PROTOCOL, COMPAT_COMPRESS, MAX_COMPRESS_LVL, PKL_EXT
from arimo.util.types.arrow import is_boolean, is_float, is_integer, is_num, is_string, string
from arimo.util.types.spark_sql import _BOOL_TYPE, _FLOAT_TYPES, _INT_TYPES, _NUM_TYPES, _STR_TYPE
import arimo.debug


_TMP_FILE_NAME = '.tmp'


_LOGGER_NAME = __name__


_UUID_ARG_NAME = 'uuid'
_TIMESTAMP_ARG_NAME = 'timestamp'


_BLUEPRINT_PARAMS_ORDERED_LIST = [
    '__BlueprintClass__',
    _UUID_ARG_NAME, '_uuid',
    _TIMESTAMP_ARG_NAME,

    'data',
        'data.id_col',
        'data.time_col',

        'data.nulls',

        'data.max_n_cats',

        'data.num_outlier_tails',
        'data.num_null_fill_method',
        'data.num_data_scaler',

        'data.repr_sample_size',

        'data.pred_vars',
        'data.pred_vars_incl',
        'data.pred_vars_excl',

        'data.force_cat',
        'data.force_cat_incl',
        'data.force_cat_excl',

        'data.force_num',
        'data.force_num_incl',
        'data.force_num_excl',

        'data._cat_prep_cols',
        'data._cat_prep_cols_metadata',

        'data._num_prep_cols',
        'data._num_prep_cols_metadata',

        'data._prep_vec_col',
        'data._prep_vec_size',

        'data._crosssect_prep_vec_size',   # *** LEGACY ***
        'data.assume_crosssect_timeser_indep',   # *** LEGACY ***

        'data._transform_pipeline_dir',

        'data.label',
            'data.label.var',
            'data.label._int_var',
            'data.label._n_classes',
            'data.label.excl_median',
            'data.label.excl_outliers',
            'data.label.outlier_tails',
            'data.label.outlier_tail_proportion',
            'data.label.lower_outlier_threshold',
            'data.label.upper_outlier_threshold',

    'min_input_ser_len',
    'max_input_ser_len',
    'pred_horizon_len',

    'model',
        'model.ver',

        'model.factory',
            'model.factory.name',

            # Arimo DL
            'model.factory.n_timeser_lstm_nodes',
            'model.factory.n_fdfwd_hid_nodes',

        'model.train',
            'model.train.objective',
            'model.train.n_samples_max_multiple_of_data_size',
            'model.train.n_samples',
            'model.train.val_proportion',
            'model.train.n_cross_val_folds',
            'model.train.n_train_samples_per_era',
            'model.train.n_train_samples_per_epoch',
            'model.train.min_n_val_samples_per_epoch',
            'model.train.batch_size',
            'model.train.val_batch_size',
            'model.train.bal_classes',

            'model.train.hyper',
                'model.train.hyper.name',

            'model.train.val_metric',
                'model.train.val_metric.name',
                'model.train.val_metric.mode',
                'model.train.val_metric.significance',

            'model.train.reduce_lr_on_plateau',
                'model.train.reduce_lr_on_plateau.patience_n_epochs',
                'model.train.reduce_lr_on_plateau.factor',

            'model.train.early_stop',
                'model.train.early_stop.patience_min_n_epochs',
                'model.train.early_stop.patience_proportion_total_n_epochs',

        'model._persist',
            'model._persist.dir',
            'model._persist.file',
            'model._persist.struct_file',
            'model._persist.weights_file',
            'model._persist.train_history_file',

        'model.score',
            'model.score.raw_score_col_prefix',

        'model.component_blueprints',

    'persist',
        'persist.local',
            'persist.local.dir_path',

        'persist.s3',
            'persist.s3.bucket',
            'persist.s3.dir_prefix',
            'persist.s3._prefix_dir_path',
            'persist.s3._dir_path',
            'persist.s3._file_key',
            'persist.s3._models_dir_prefix',
            'persist.s3._models_dir_path',

        'persist._file',
        'persist._models_dir'
    ]


def _docstr_blueprint(BlueprintClass):
    s = '**BLUEPRINT PARAMS** (``<blueprint_instance>.params``): a ``dict`` of below default configurations, plus any other custom configurations you would like to set:\n'

    for blueprint_param_key in \
            sorted(
                [k for k in BlueprintClass._DEFAULT_PARAMS.keys(all_nested=True)
                   if not (k.startswith('_') or ('._' in k))],
                key=_BLUEPRINT_PARAMS_ORDERED_LIST.index):
        blueprint_param_value = \
            BlueprintClass._DEFAULT_PARAMS[blueprint_param_key]

        blueprint_param_metadata = \
            BlueprintClass._DEFAULT_PARAMS(blueprint_param_key)

        blueprint_param_label = \
            blueprint_param_metadata.get('label')

        blueprint_param_description = \
            blueprint_param_metadata.get('description')

        if not isinstance(blueprint_param_value, (dict, Namespace)):
            s += '\n- ``{0}``: `{1}`{2}\n'.format(
                blueprint_param_key,
                blueprint_param_value,
                (u' — {0} `({1})`'.format(blueprint_param_label, blueprint_param_description)
                 if blueprint_param_description
                 else u' — {0}'.format(blueprint_param_label))
                if blueprint_param_label
                else '')

    BlueprintClass.__doc__ = \
        ('' if BlueprintClass.__doc__ is None
            else BlueprintClass.__doc__) \
        + '\n{0}\n'.format(s)

    return BlueprintClass


@_docstr_blueprint
class _BlueprintABC(object):
    """
    Abstract base class for ``Blueprint``s
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def __repr__(self):
        raise NotImplementedError

    _MODE_ARG = '__mode__'
    _TRAIN_MODE = 'train'
    _SCORE_MODE = 'score'
    _EVAL_MODE = 'eval'

    _PREP_ADF_ALIAS_SUFFIX = '__Prep'

    _GLOBAL_EVAL_KEY = 'GLOBAL'
    _BY_ID_EVAL_KEY = 'BY_ID'

    _MODEL_PATHS_ON_SPARK_WORKER_NODES = {}

    @abc.abstractmethod
    def prep_data(self, df, **kwargs):
        """
        Required method to prepare / pre-process business data into normalized numerical data for AI modeling
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, df, **kwargs):
        """
        Required method to (re)train AI model(s)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def score(self, df, **kwargs):
        """
        Required method to score new data using trained AI model(s)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, df, **kwargs):
        """
        Required method to evaluate AI model metrics on labeled data
        """
        raise NotImplementedError

    _DEFAULT_PARAMS = \
        Namespace(
            __BlueprintClass__=None,

            uuid=None,

            timestamp=None,

            data=Namespace(
                # index columns
                id_col=None,
                time_col=None,

                # NULLs
                nulls=Namespace(),

                # cat & num data treatments
                max_n_cats=9,

                num_outlier_tails='both',
                num_null_fill_method='avg',
                num_data_scaler='standard',

                # repr sample size
                repr_sample_size=10 ** 6,

                # treatments of cat & num vars
                force_cat=None,
                force_cat_incl=_PRED_VARS_INCL_T_CAT_AUX_COLS,
                force_cat_excl=None,

                force_num=None,
                force_num_incl=_PRED_VARS_INCL_T_NUM_AUX_COLS,
                force_num_excl=None,

                # directory to which to persist data transformation pipeline
                _transform_pipeline_dir='DataTransforms'),

            model=Namespace(
                ver=None),

            persist=Namespace(
                local=Namespace(
                    dir_path='/tmp/.arimo/blueprints'),

                s3=Namespace(
                    bucket=None,
                    dir_prefix='.arimo/blueprints'),

                _file='Blueprint' + PKL_EXT,

                _models_dir='Models'),

            __metadata__={
                '__BlueprintClass__': Namespace(),

                _UUID_ARG_NAME: Namespace(),

                _TIMESTAMP_ARG_NAME: Namespace(),

                'data': Namespace(
                    label='Data Params'),

                'data.id_col': Namespace(
                    label='Identity/Entity Column',
                    description='Name of Column Containing Identities/Entities of Predictive Interest',
                    type='string',
                    default=None),

                'data.time_col': Namespace(
                    label='Date/Time Column',
                    description='Name of Column Containing Dates/Timestamps',
                    type='string',
                    default=None),

                'data.nulls': Namespace(),

                'data.max_n_cats': Namespace(
                    label='Max No. of Categorical Levels',
                    description='Max No. of Levels Considered for Valid Categorical Variables',
                    type='int',
                    default=12   # Month of Year is probably most numerous-category cat var
                ),

                'data.num_outlier_tails': Namespace(
                    label='Numerical Data Outlier Tails',
                    description='Tail(s) of Distributions of Numerical Variables in Which to Watch for Outliers',
                    type='string',
                    choices=['both', 'lower', 'upper'],
                    default='both'),

                'data.num_null_fill_method': Namespace(
                    label='Numerical Data NULL-Filling Method',
                    description='Method to Fill NULL/NaN Values in Numerical Variables',
                    type='string',
                    choices=['avg', 'min', 'max',
                             'avg_before', 'min_before', 'max_before',
                             'avg_after', 'min_after', 'max_after'],
                    default='avg'),

                'data.num_data_scaler': Namespace(
                    label='Numerical Data Scaler',
                    description='Scaling Method for Numerical Variables',
                    type='string',
                    choices=[None, 'standard', 'maxabs', 'minmax'],
                    default='standard'),

                'data.repr_sample_size': Namespace(),

                'data.force_cat': Namespace(
                    label='Forced Categorical Variables',
                    description='Variables Forced to be Interpreted as Categorical',
                    type='list',
                    default=[]),

                'data.force_cat_incl': Namespace(),

                'data.force_cat_excl': Namespace(),

                'data.force_num': Namespace(
                    label='Forced Numerical Variables',
                    description='Variables Forced to be Interpreted as Numerical',
                    type='list',
                    default=[]),

                'data.force_num_incl': Namespace(),

                'data.force_num_excl': Namespace(),

                'data._transform_pipeline_dir': Namespace(),

                'model': Namespace(
                    label='Model Params'),

                'model.ver': Namespace(
                    label='Initial Model Version',
                    description='Version of Model Used for Fresh Training, Incremental Training or Scoring',
                    type='string',
                    default=None),

                'persist': Namespace(),
                
                'persist.local': Namespace(),
                
                'persist.local.dir_path': Namespace(),

                'persist.s3': Namespace(),

                'persist.s3.bucket': Namespace(),

                'persist.s3.dir_prefix': Namespace(),

                'persist._file': Namespace(),

                'persist._models_dir': Namespace()})

    @classmethod
    def __qual_name__(cls):
        return '{}.{}'.format(cls.__module__, cls.__name__)

    @classmethod
    def class_logger(cls, *handlers, **kwargs):
        logger = logging.getLogger(name=cls.__qual_name__())

        level = kwargs.get('level')

        if level is None:
            level = logging.DEBUG \
                if arimo.debug.ON \
                else logging.INFO

        logger.setLevel(level)

        if kwargs.get('verbose'):
            handlers += (STDOUT_HANDLER,)

        for handler in handlers:
            logger.addHandler(handler)

        return logger

    @classmethod
    def class_stdout_logger(cls):
        return cls.class_logger(
            level=logging.DEBUG,
            verbose=True)

    def logger(self, *handlers, **kwargs):
        logger = logging.getLogger(name=str(self))

        level = kwargs.get('level')

        if level is None:
            level = logging.DEBUG \
                if arimo.debug.ON \
                else logging.INFO

        logger.setLevel(level)

        if kwargs.get('verbose'):
            handlers += (STDOUT_HANDLER,)

        for handler in handlers:
            logger.addHandler(handler)

        return logger

    @property
    def stdout_logger(self):
        return self.logger(
            level=logging.DEBUG,
            verbose=True)

    def __init__(self, params={}, aws_access_key_id=None, aws_secret_access_key=None, verbose=False, **kwargs):
        # set up authentication
        self.auth = Namespace(
            aws=Namespace(
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key))

        # get Blueprint params
        self.params = copy.deepcopy(self._DEFAULT_PARAMS)
        self.params.update(params, **kwargs)

        # make local dir to store content of Blueprints & Blueprinted models
        fs.mkdir(
            dir=self.params.persist.local.dir_path,
            hdfs=False)

        # if UUID is specified, then load the corresponding Blueprint from local
        if self.params.uuid:
            local_file_path = \
                os.path.join(
                    self.params.persist.local.dir_path,
                    self.params.uuid,
                    self.params.persist._file)

            if os.path.isfile(local_file_path):
                existing_params = joblib.load(filename=local_file_path)
                del existing_params.uuid
                self.params.update(existing_params)

                create_blueprint = False

            else:
                create_blueprint = True
                create_blueprint_w_uuid = self.params.uuid

        else:
            create_blueprint = True
            create_blueprint_w_uuid = None

        if create_blueprint:
            # if UUID is None, generate UUID for the Blueprint
            self.params.uuid = \
                str(uuid.uuid4()) \
                if create_blueprint_w_uuid is None \
                else create_blueprint_w_uuid

            self.params._uuid = clean_uuid(self.params.uuid)

            # save the full identifier of the Blueprint Class for subsequent loading
            self.params.__BlueprintClass__ = self.__qual_name__()

        # set local dir path for Blueprint
        self.dir = \
            os.path.join(
                self.params.persist.local.dir_path,
                self.params.uuid)

        fs.mkdir(
            dir=self.dir,
            hdfs=False)

        # set local file path for Blueprint
        self.file = \
            os.path.join(
                self.dir,
                self.params.persist._file)

        # set local dir storing data transforms for Blueprint
        self.data_transforms_dir = \
            os.path.join(
                self.dir,
                self.params.data._transform_pipeline_dir)

        # print Blueprint params if in verbose mode
        if verbose:
            self.stdout_logger.info('\n{}'.format(self.params))

    def __str__(self):
        return repr(self)

    @property
    def _persist_on_s3(self):   # generate helper attributes if to save Blueprint & Blueprinted Models on S3
        if self.params.persist.s3.bucket and self.params.persist.s3.dir_prefix:
            # set up Boto3 S3 client
            self._s3_client = \
                s3.client(
                    access_key_id=self.auth.aws.access_key_id,
                    secret_access_key=self.auth.aws.secret_access_key)

            # full Prefix Dir Path combining S3 Bucket Name & Dir Prefix
            self.params.persist.s3._prefix_dir_path = \
                's3://{}/{}'.format(
                    self.params.persist.s3.bucket,
                    self.params.persist.s3.dir_prefix)

            # full Blueprint Dir Path combining Project Dir Path & Blueprint UUID
            self.params.persist.s3._dir_path = \
                os.path.join(
                    self.params.persist.s3._prefix_dir_path,
                    self.params.uuid)

            # full path of Blueprint file within S3 Bucket
            self.params.persist.s3._file_key = \
                os.path.join(
                    self.params.persist.s3.dir_prefix,
                    self.params.uuid,
                    self.params.persist._file)

            # full path of S3 dir containing data transforms
            self.params.persist.s3._data_transforms_dir_path = \
                os.path.join(
                    self.params.persist.s3._dir_path,
                    self.params.data._transform_pipeline_dir)

            # path of data transforms within S3 Bucket
            self.params.persist.s3._data_transforms_dir_prefix = \
                os.path.join(
                    self.params.persist.s3.dir_prefix,
                    self.params.uuid,
                    self.params.data._transform_pipeline_dir)

            # full path of S3 dir containing models
            self.params.persist.s3._models_dir_path = \
                os.path.join(
                    self.params.persist.s3._dir_path,
                    self.params.persist._models_dir)

            # path of models dir within S3 Bucket
            self.params.persist.s3._models_dir_prefix = \
                os.path.join(
                    self.params.persist.s3.dir_prefix,
                    self.params.uuid,
                    self.params.persist._models_dir)

            return True

        else:
            return False

    @property
    def path(self):
        return self.params.persist.s3._dir_path \
            if self._persist_on_s3 \
            else self.dir

    def save(self, verbose=True):
        # update timestamp
        self.params[_TIMESTAMP_ARG_NAME] = \
            date_time.now(
                strf='%Y-%m-%d %H:%M:%S',
                utc=True)

        if verbose:
            msg = 'Saving to Local Path "{}"...'.format(self.file)
            self.stdout_logger.info(msg)
            tic = time.time()

        joblib.dump(
            self.params,
            filename=self.file,
            compress=(COMPAT_COMPRESS, MAX_COMPRESS_LVL),
            protocol=COMPAT_PROTOCOL)

        if verbose:
            toc = time.time()
            self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

        if self._persist_on_s3:
            if verbose:
                msg = 'Saving to S3 Path "s3://{}/{}"...'.format(
                    self.params.persist.s3.bucket, self.params.persist.s3._file_key)
                self.stdout_logger.info(msg)
                tic = time.time()

            self._s3_client.upload_file(
                Filename=self.file,
                Bucket=self.params.persist.s3.bucket,
                Key=self.params.persist.s3._file_key)

            if os.path.isdir(self.data_transforms_dir):
                s3.sync(
                    from_dir_path=self.data_transforms_dir,
                    to_dir_path=self.params.persist.s3._data_transforms_dir_path,
                    access_key_id=self.auth.aws.access_key_id,
                    secret_access_key=self.auth.aws.secret_access_key,
                    delete=True, quiet=False,
                    verbose=verbose)

            if verbose:
                toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

    def copy(self, uuid, verbose=True, **kwargs):
        params = copy.deepcopy(self.params)
        params.uuid = uuid

        aws_access_key_id = kwargs.pop('aws_access_key_id', self.auth.aws.access_key_id)
        aws_secret_access_key = kwargs.pop('aws_secret_access_key', self.auth.aws.secret_access_key)

        blueprint = \
            type(self)(
                params=params,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                verbose=verbose,
                **kwargs)

        if self._persist_on_s3 or blueprint._persist_on_s3:
            s3.sync(
                from_dir_path=self.path,
                to_dir_path=blueprint.path,
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key,
                delete=True, quiet=not verbose,
                verbose=verbose)

        else:
            fs.cp(
                from_path=self.path,
                to_path=blueprint.path,
                hdfs=False,
                is_dir=True)

        blueprint.save(verbose=verbose)

        return blueprint


class _BlueprintedModelABC(object):
    """
    Blueprinted Model abstract base class
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def load(self, verbose=True):
        """
        Required method to load model object
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, verbose=True):
        """
        Required method to save model object
        """
        raise NotImplementedError

    @classmethod
    def __qual_name__(cls):
        return '{}.{}'.format(cls.__module__, cls.__name__)

    def __repr__(self):
        return '{} {} ({}) v{}'.format(
            self.blueprint,
            type(self).__name__,
            self.blueprint.params.model.factory.name,
            self.ver)

    def __str__(self):
        return repr(self)

    def logger(self, *handlers, **kwargs):
        logger = logging.getLogger(name=str(self))

        level = kwargs.get('level')

        if level is None:
            level = logging.DEBUG \
                if arimo.debug.ON \
                else logging.INFO

        logger.setLevel(level)

        if kwargs.get('verbose'):
            handlers += (STDOUT_HANDLER,)

        for handler in handlers:
            logger.addHandler(handler)

        return logger

    @property
    def stdout_logger(self):
        return self.logger(
            level=logging.DEBUG,
            verbose=True)

    def __init__(self, blueprint, ver=None, verbose=True):
        global _LOADED_BLUEPRINTS

        self.blueprint = blueprint

        if (blueprint.path not in _LOADED_BLUEPRINTS) and \
                (blueprint._persist_on_s3) and \
                ('Contents' in \
                    blueprint._s3_client.list_objects_v2(
                        Bucket=blueprint.params.persist.s3.bucket,
                        Prefix=blueprint.params.persist.s3._models_dir_prefix)):
            if verbose:
                msg = 'Downloading All Models Trained by {} from S3 Path "{}"...'.format(
                    blueprint, blueprint.params.persist.s3._models_dir_path)
                blueprint.stdout_logger.info(msg)

            s3.sync(
                from_dir_path=blueprint.params.persist.s3._models_dir_path,
                to_dir_path=blueprint.models_dir,
                access_key_id=blueprint.auth.aws.access_key_id,
                secret_access_key=blueprint.auth.aws.secret_access_key,
                delete=True, quiet=False)

            if verbose:
                blueprint.stdout_logger.info(msg + ' done!')

        # force ver to be string, just in case it's fed in as integer
        if ver is not None:
            ver = str(ver)

        # set Model Ver
        self.ver = \
            (max(os.listdir(blueprint.models_dir))
             if ver.lower() == 'latest'
             else ver) \
            if ver \
            else date_time.now(strf='%Y%m%d%H%M%S', utc=True)

        # Model local dir path, combining Blueprints' Models Dir Path & Model Ver
        self.dir = \
            os.path.join(
                blueprint.models_dir,
                self.ver)

        fs.mkdir(
            dir=self.dir,
            hdfs=False)

        # initiate empty Model object
        self._obj = None

        # load existing model if version is not new
        if ver is not None:
            self.load()

    def __getattr__(self, item):   # if cannot resolve item in the MRO, then access item via _obj
        if not self._obj:   # if model object is not existing, initiate with Blueprint's Model Factory
            model_factory_params = copy.deepcopy(self.blueprint.params.model.factory)

            if isinstance(model_factory_params, argparse.Namespace):
                model_factory_params = model_factory_params.__dict__

            model_factory_params.pop('__metadata__')

            model_factory_name = model_factory_params.pop('name')

            msg = 'Initializing by {}...'.format(model_factory_name)
            self.stdout_logger.info(msg)

            model_factory = import_obj(model_factory_name)

            if model_factory_name.startswith('arimo.dl.experimental.keras'):
                model_factory_params['params'] = self.blueprint.params

            self._obj = model_factory(**model_factory_params)

            self.stdout_logger.info(msg + ' done!')

        return getattr(self._obj, item)

    def copy(self):
        # create a new version with Blueprint's Model Factory
        model = type(self)(
            blueprint=self.blueprint,
            ver=None)

        # use existing model object to overwrite recipe's initial model
        model._obj = self._obj

        return model


class BlueprintedArimoDLModel(_BlueprintedModelABC):
    _LOADED_MODELS = {}

    def load(self, verbose=True):
        if self.dir in self._LOADED_MODELS:
            self._obj = self._LOADED_MODELS[self.dir]

        elif os.path.isdir(self.dir):
            if verbose:
                msg = 'Loading Model from Local Directory {}...'.format(self.dir)
                self.stdout_logger.info(msg)

            self._obj = ModelServingPersistence.load(path=self.dir).model

            if verbose:
                self.stdout_logger.info(msg + ' done!')

        elif verbose:
            self.stdout_logger.info(
                'No Existing Model Object to Load at "{}"'
                    .format(self.dir))

    def save(self, verbose=True):
        # save Blueprint
        self.blueprint.save(verbose=verbose)

        if verbose:
            message = 'Saving Model to Local Directory {}...'.format(self.dir)
            self.stdout_logger.info(message + '\n')

        ModelServingPersistence(
            model=self._obj,
            preprocessor=
                DataFramePreprocessor(
                    feature_cols=self.blueprint.params.data._cat_prep_cols + self.blueprint.params.data._num_prep_cols,
                    target_col=self.blueprint.params.data.label.var,
                    num_targets=1,
                    embedding_col=None,
                    normalization=None),
            extra_artifacts=None) \
        .save(path=self.dir)

        if self.blueprint._persist_on_s3:
            if verbose:
                msg = 'Uploading All Trained Models to S3 Path "{}"...'.format(self.blueprint.params.persist.s3._models_dir_path)
                self.blueprint.stdout_logger.info(msg)

            s3.sync(
                from_dir_path=self.blueprint.models_dir,
                to_dir_path=self.blueprint.params.persist.s3._models_dir_path,
                access_key_id=self.blueprint.auth.aws.access_key_id,
                secret_access_key=self.blueprint.auth.aws.secret_access_key,
                delete=False,   # to allow multiple training jobs to upload new models to S3 at same time
                quiet=False)

            if verbose:
                self.blueprint.stdout_logger.info(msg + ' done!')

        if verbose:
            self.stdout_logger.info(message + ' done!')


class BlueprintedKerasModel(_BlueprintedModelABC):
    _LOADED_MODELS = {}

    def load(self, verbose=True):
        local_file_path = \
            os.path.join(
                self.dir,
                self.blueprint.params.model._persist.file)

        if local_file_path in self._LOADED_MODELS:
            self._obj = self._LOADED_MODELS[local_file_path]

        elif os.path.isfile(local_file_path):
            if verbose:
                msg = 'Loading Model from Local Directory {}...'.format(self.dir)
                self.stdout_logger.info(msg)

            self._obj = self._LOADED_MODELS[local_file_path] = \
                arimo.backend.keras.models.load_model(
                    filepath=local_file_path)

            if verbose:
                self.stdout_logger.info(msg + ' done!')

        elif verbose:
            self.stdout_logger.info(
                'No Existing Model Object to Load at "{}"'
                    .format(local_file_path))

    def save(self, verbose=True):
        # save Blueprint
        self.blueprint.save(verbose=verbose)

        if verbose:
            message = 'Saving Model to Local Directory {}...'.format(self.dir)
            self.stdout_logger.info(message + '\n')

        self._obj.save(
            filepath=os.path.join(
                self.dir,
                self.blueprint.params.model._persist.file),
            overwrite=True,
            include_optimizer=True)

        joblib.dump(
            self.history,
            filename=os.path.join(
                self.dir,
                self.blueprint.params.model._persist.train_history_file),
            compress=(COMPAT_COMPRESS, MAX_COMPRESS_LVL),
            protocol=COMPAT_PROTOCOL)

        if self.blueprint._persist_on_s3:
            if verbose:
                msg = 'Uploading All Trained Models to S3 Path "{}"...'.format(self.blueprint.params.persist.s3._models_dir_path)
                self.blueprint.stdout_logger.info(msg)

            s3.sync(
                from_dir_path=self.blueprint.models_dir,
                to_dir_path=self.blueprint.params.persist.s3._models_dir_path,
                access_key_id=self.blueprint.auth.aws.access_key_id,
                secret_access_key=self.blueprint.auth.aws.secret_access_key,
                delete=False,   # to allow multiple training jobs to upload new models to S3 at same time
                quiet=False)

            if verbose:
                self.blueprint.stdout_logger.info(msg + ' done!')

        if verbose:
            self.stdout_logger.info(message + ' done!')


class _TimeSerDataPrepMixInABC(object):
    pass


class _EvalMixInABC(object):
    __metaclass__ = abc.ABCMeta

    _EVAL_ADF_ALIAS_SUFFIX = '__Eval'

    @classmethod
    @abc.abstractproperty
    def eval_metrics(cls):
        raise NotImplementedError


class ClassifEvalMixIn(_EvalMixInABC):
    eval_metrics = \
        'Prevalence', 'ConfMat', 'Acc', \
        'Precision', 'WeightedPrecision', \
        'Recall', 'WeightedRecall', \
        'F1', 'WeightedF1', \
        'PR_AuC', 'Weighted_PR_AuC', \
        'ROC_AuC', 'Weighted_ROC_AuC'


class RegrEvalMixIn(_EvalMixInABC):
    eval_metrics = 'MedAE', 'MAE', 'RMSE', 'R2'


@_docstr_blueprint
class _SupervisedBlueprintABC(_BlueprintABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_PARAMS = \
        copy.deepcopy(
            _BlueprintABC._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        data=Namespace(
            # pred vars
            pred_vars=None,   # None implies considering all vars
            pred_vars_incl=_PRED_VARS_INCL_T_AUX_COLS,
            pred_vars_excl=None,

            _cat_prep_cols=(),
            _cat_prep_cols_metadata={},

            _num_prep_cols=(),
            _num_prep_cols_metadata=(),

            _prep_vec_col='__Xvec__',
            _prep_vec_size=None,

            _crosssect_prep_vec_size=0,   # *** LEGACY ***
            assume_crosssect_timeser_indep=True,    # *** LEGACY ***

            # label
            label=Namespace(
                var='y',
                _int_var=None,
                _n_classes=None,
                excl_median=None,
                excl_outliers=True,
                outlier_tails='both',
                outlier_tail_proportion=5e-3,
                lower_outlier_threshold=None,
                upper_outlier_threshold=None)),

        model=Namespace(
            factory=Namespace(
                name=None),
            train=Namespace(
                objective=None,
                n_samples=248 * 10 ** 6,
                val_proportion=.32,
                n_cross_val_folds=3,
                hyper=Namespace(
                    name='HyperOpt')),
            _persist=Namespace(),
            score=Namespace(
                raw_score_col_prefix='__RawPred__')),

        __metadata__={
            'data': Namespace(
                label='Data Params'),

            'data.pred_vars': Namespace(
                label='Predictor Variables',
                description='List of Predictor Variables to Consider (default: [], meaning all variables)',
                type='list',
                default=[]),

            'data.pred_vars_incl': Namespace(
                label='Predictor Variables Include',
                description='List of Predictor Variables to Include',
                type='list',
                default=[]),

            'data.pred_vars_excl': Namespace(
                label='Predictor Variables Exclude',
                description='List of Predictor Variables to Exclude',
                type='list',
                default=[]),

            'data._cat_prep_cols': Namespace(),

            'data._cat_prep_cols_metadata': Namespace(),

            'data._num_prep_cols': Namespace(),

            'data._num_prep_cols_metadata': Namespace(),

            'data._prep_vec_col': Namespace(),

            'data._prep_vec_size': Namespace(),

            'data._crosssect_prep_vec_size': Namespace(),   # *** LEGACY **

            'data.assume_crosssect_timeser_indep': Namespace(),   # *** LEGACY **

            'data.label': Namespace(
                label='Target-Labeling Params'),

            'data.label.var': Namespace(
                label='Target Label Column to Predict',
                description='Column Containing Target Labels to Predict',
                type='string',
                default='y'),

            'data.label._int_var': Namespace(),

            'data.label._n_classes': Namespace(),

            'data.label.excl_median': Namespace(),

            'data.label.excl_outliers': Namespace(),

            'data.label.outlier_tails': Namespace(),

            'data.label.outlier_tail_proportion': Namespace(),

            'data.label.lower_outlier_threshold': Namespace(),

            'data.label.upper_outlier_threshold': Namespace(),

            'model': Namespace(
                label='Model Params'),

            'model.factory': Namespace(
                label='Model-Initializing Factory Function Name & Params'),

            'model.factory.name': Namespace(
                label='Model Factory Function',
                description='Full Name package.module.submodule.factory_func_name',
                type='string',
                default=None),

            'model.train': Namespace(
                label='Model-Training Params'),

            'model.train.objective': Namespace(
                label='Objective Function',
                description='Model Training Objective Function',
                type='string',
                default=None),

            'model.train.n_samples': Namespace(
                label='No. of Training Samples',
                description='No. of Data Samples to Use for Training',
                type='int',
                default=1000000),

            'model.train.val_proportion': Namespace(
                label='Validation Data Proportion',
                description='Proportion of Training Data Samples to Use for Validating Model during Training',
                type='float',
                default=.32),

            'model.train.n_cross_val_folds': Namespace(
                label='No. of Cross-Validation Folds',
                description='No. of Cross-Validation Folds',
                type='int',
                default=3),

            'model.train.hyper': Namespace(),

            'model.train.hyper.name': Namespace(
                label='Hyper-Parameter Optimization Mechanism',
                description='Hyper-Parameter Optimization Mechanism',
                type='string',
                choices=['Grid', 'SciPy',
                         'BayesOpt', 'HPOlib', 'HyperOpt', 'MOE', 'OpenTuner', 'Optunity', 'Spearmint'],
                default='HyperOpt'),

            'model._persist': Namespace(),

            'model.score': Namespace(),

            'model.score.raw_score_col_prefix': Namespace()})

    _INT_LABEL_COL = '__INT_LABEL__'
    _LABELED_ADF_ALIAS_SUFFIX = '__Labeled'

    def __init__(self, *args, **kwargs):
        super(_SupervisedBlueprintABC, self).__init__(*args, **kwargs)

        # set local dir storing all models for Blueprint
        if self.params.persist._models_dir != self._DEFAULT_PARAMS.persist._models_dir:
            self.params.persist._models_dir = self._DEFAULT_PARAMS.persist._models_dir

        self.models_dir = \
            os.path.join(
                self.dir,
                self.params.persist._models_dir)

        # set __BlueprintedModelClass__
        self.__BlueprintedModelClass__ = \
            BlueprintedKerasModel \
            if self.params.model.factory.name.startswith('arimo.dl.experimental.keras') \
            else BlueprintedArimoDLModel

        self.params.__BlueprintedModelClass__ = \
            self.__BlueprintedModelClass__.__qual_name__()

    def __repr__(self):
        return '{} Instance "{}" (Label: "{}")'.format(
            self.__qual_name__(),
            self.params.uuid,
            self.params.data.label.var)

    def prep_data(
            self, df, __mode__='score',
            __from_ppp__=False,
            __ohe_cat__=False, __scale_cat__=True, __vectorize__=True,
            verbose=True, **kwargs):
        # check if (incrementally-)training, scoring or evaluating
        if __mode__ == self._TRAIN_MODE:
            __train__ = True
            __score__ = __eval__ = False

        elif __mode__ == self._SCORE_MODE:
            __train__ = __eval__ = False
            __score__ = True

        elif __mode__ == self._EVAL_MODE:
            __train__ = __score__ = False
            __eval__ = True

        else:
            raise ValueError(
                '*** Blueprint "{}" argument must be either "{}", "{}" or "{}" ***'
                    .format(self._MODE_ARG, self._TRAIN_MODE, self._SCORE_MODE, self._EVAL_MODE))

        assert __train__ + __score__ + __eval__ == 1

        __first_train__ = __train__ and (not os.path.isdir(self.data_transforms_dir))

        if isinstance(df, SparkADF):
            adf = df

            adf.tCol = self.params.data.time_col

            if isinstance(self, _TimeSerDataPrepMixInABC):
                adf.iCol = self.params.data.id_col

        else:
            kwargs['tCol'] = self.params.data.time_col

            if isinstance(self, _TimeSerDataPrepMixInABC):
                kwargs['iCol'] = self.params.data.id_col

            if isinstance(df, pandas.DataFrame):
                adf = SparkADF.create(
                        data=df,
                        schema=None,
                        samplingRatio=None,
                        verifySchema=False,
                        **kwargs)

            elif isinstance(df, pyspark.sql.DataFrame):
                adf = SparkADF(sparkDF=df, **kwargs)

            else:
                __vectorize__ = False

                if isinstance(df, ArrowADF):
                    adf = df

                    adf.tCol = self.params.data.time_col

                    if isinstance(self, _TimeSerDataPrepMixInABC):
                        adf.iCol = self.params.data.id_col

                else:
                    assert isinstance(df, _STR_CLASSES)

                    adf = (ArrowSparkADF
                           if arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS
                           else ArrowADF)(
                        path=df, **kwargs)

        if __train__:
            assert self._INT_LABEL_COL not in adf.columns

            label_col_type = adf.type(self.params.data.label.var)

            if isinstance(adf, ArrowADF):
                sample_label_series = None

                if is_float(label_col_type) and isinstance(self, ClassifEvalMixIn):
                    adf.map(
                        mapper=_ArrowADF__castType__pandasDFTransform(
                                col=self.params.data.label.var,
                                asType=str,
                                asCol=None),
                        inheritNRows=True,
                        inplace=True)

                    label_col_type = string()

                if is_boolean(label_col_type):
                    if __first_train__:
                        self.params.data.label._int_var = self._INT_LABEL_COL

                    adf.map(
                        mapper=_ArrowADF__castType__pandasDFTransform(
                                col=self.params.data.label.var,
                                asType=int,
                                asCol=self.params.data.label._int_var),
                        inheritNRows=True,
                        inplace=True)

                elif is_string(label_col_type):
                    if __first_train__:
                        self.params.data.label._int_var = self._INT_LABEL_COL

                        if sample_label_series is None:
                            sample_label_series = \
                                adf.copy(
                                    resetMappers=True,
                                    inheritCache=True,
                                    inheritNRows=True
                                ).sample(
                                    self.params.data.label.var,
                                    n=10 ** 8   # 1mil = 68MB
                                )[self.params.data.label.var]

                            sample_label_series = \
                                sample_label_series.loc[
                                    pandas.notnull(sample_label_series)]

                        self.params.data.label._strings = \
                            LabelEncoder() \
                            .fit(sample_label_series) \
                            .classes_.tolist()

                    adf.map(
                        mapper=_ArrowADF__encodeStr__pandasDFTransform(
                                col=self.params.data.label.var,
                                strs=self.params.data.label._strings,
                                asCol=self.params.data.label._int_var),
                        inheritNRows=True,
                        inplace=True)

                elif is_integer(label_col_type) and __first_train__:
                    self.params.data.label._int_var = self.params.data.label.var

                if is_num(label_col_type) and isinstance(self, RegrEvalMixIn):
                    assert self.params.data.label.excl_outliers \
                       and self.params.data.label.outlier_tails \
                       and self.params.data.label.outlier_tail_proportion \
                       and (self.params.data.label.outlier_tail_proportion < .5)

                    if __first_train__:
                        lower_numeric_null, upper_numeric_null = \
                            self.params.data.nulls.get(
                                self.params.data.label.var,
                                (None, None))

                        self.params.data.label.outlier_tails = \
                            self.params.data.label.outlier_tails.lower()

                        _calc_lower_outlier_threshold = \
                            (self.params.data.label.outlier_tails in ('both', 'lower')) and \
                            pandas.isnull(self.params.data.label.lower_outlier_threshold)

                        _calc_upper_outlier_threshold = \
                            (self.params.data.label.outlier_tails in ('both', 'upper')) and \
                            pandas.isnull(self.params.data.label.upper_outlier_threshold)

                        if _calc_lower_outlier_threshold:
                            if sample_label_series is None:
                                sample_label_series = \
                                    adf.copy(
                                        resetMappers=True,
                                        inheritCache=True,
                                        inheritNRows=True
                                    ).sample(
                                        self.params.data.label.var,
                                        n=10 ** 8   # 1mil = 68MB
                                    )[self.params.data.label.var]

                                sample_label_series = \
                                    sample_label_series.loc[
                                        pandas.notnull(sample_label_series) &
                                        numpy.isfinite(sample_label_series)]

                            if _calc_upper_outlier_threshold:
                                self.params.data.label.lower_outlier_threshold, \
                                self.params.data.label.upper_outlier_threshold = \
                                    sample_label_series.quantile(
                                        q=(self.params.data.label.outlier_tail_proportion,
                                           1 - self.params.data.label.outlier_tail_proportion),
                                        interpolation='linear')

                                if (lower_numeric_null is not None) and \
                                        (lower_numeric_null > self.params.data.label.lower_outlier_threshold):
                                    assert lower_numeric_null < self.params.data.label.upper_outlier_threshold, \
                                        '*** {} >= {} ***'.format(
                                            lower_numeric_null,
                                            self.params.data.label.upper_outlier_threshold)

                                    self.params.data.label.lower_outlier_threshold = lower_numeric_null

                                if (upper_numeric_null is not None) and \
                                        (upper_numeric_null < self.params.data.label.upper_outlier_threshold):
                                    assert upper_numeric_null > self.params.data.label.lower_outlier_threshold, \
                                        '*** {} <= {} ***'.format(
                                            upper_numeric_null,
                                            self.params.data.label.lower_outlier_threshold)

                                    self.params.data.label.upper_outlier_threshold = upper_numeric_null

                            else:
                                self.params.data.label.lower_outlier_threshold, \
                                M = sample_label_series.quantile(
                                        q=(self.params.data.label.outlier_tail_proportion,
                                           1),
                                        interpolation='linear')

                                if (lower_numeric_null is not None) and \
                                        (lower_numeric_null > self.params.data.label.lower_outlier_threshold):
                                    assert lower_numeric_null < M, \
                                        '*** {} >= {} ***'.format(lower_numeric_null, M)

                                    self.params.data.label.lower_outlier_threshold = lower_numeric_null

                        elif _calc_upper_outlier_threshold:
                            if sample_label_series is None:
                                sample_label_series = \
                                    adf.copy(
                                        resetMappers=True,
                                        inheritCache=True,
                                        inheritNRows=True
                                    ).sample(
                                        self.params.data.label.var,
                                        n=10 ** 8   # 1mil = 68MB
                                    )[self.params.data.label.var]

                                sample_label_series = \
                                    sample_label_series.loc[
                                        pandas.notnull(sample_label_series) &
                                        numpy.isfinite(sample_label_series)]

                            m, self.params.data.label.upper_outlier_threshold = \
                                sample_label_series.quantile(
                                    q=(0,
                                       1 - self.params.data.label.outlier_tail_proportion),
                                    interpolation='linear')

                            if (upper_numeric_null is not None) and \
                                    (upper_numeric_null < self.params.data.label.upper_outlier_threshold):
                                assert upper_numeric_null > m, \
                                    '*** {} <= {} ***'.format(upper_numeric_null, m)

                                self.params.data.label.upper_outlier_threshold = upper_numeric_null

            else:
                if isinstance(adf, ArrowSparkADF):
                    __vectorize__ = False

                if __from_ppp__:
                    assert adf.alias

                else:
                    adf_uuid = clean_uuid(uuid.uuid4())

                    adf.alias = \
                        '{}__{}__{}'.format(
                            self.params._uuid,
                            __mode__,
                            adf_uuid)

                if (label_col_type.startswith('decimal') or (label_col_type in _FLOAT_TYPES)) \
                        and isinstance(self, ClassifEvalMixIn):
                    adf('STRING({0}) AS {0}'.format(self.params.data.label.var),
                        *(col for col in adf.columns
                          if col != self.params.data.label.var),
                        inheritCache=True,
                        inheritNRows=True,
                        inplace=True)

                    label_col_type = _STR_TYPE

                if label_col_type == _BOOL_TYPE:
                    if __first_train__:
                        self.params.data.label._int_var = self._INT_LABEL_COL

                    adf('*',
                        'INT({}) AS {}'.format(
                            self.params.data.label.var,
                            self.params.data.label._int_var),
                        inheritCache=True,
                        inheritNRows=True,
                        inplace=True)

                elif label_col_type == _STR_TYPE:
                    if __first_train__:
                        self.params.data.label._int_var = self._INT_LABEL_COL

                        self.params.data.label._strings = \
                            [s for s in
                                StringIndexer(
                                    inputCol=self.params.data.label.var,
                                    outputCol=self.params.data.label._int_var,
                                    handleInvalid='skip'   # filter out rows with invalid data (just in case there are NULL labels)
                                        # 'error': throw an error
                                        # 'keep': put invalid data in a special additional bucket, at index numLabels
                                ).fit(dataset=adf).labels
                             if pandas.notnull(s)]

                    adf('*',
                        '(CASE {} ELSE NULL END) AS {}'.format(
                            ' '.join(
                                "WHEN {} = '{}' THEN {}".format(
                                    self.params.data.label.var, label, i)
                                for i, label in enumerate(self.params.data.label._strings)),
                            self.params.data.label._int_var),
                        inheritCache=True,
                        inheritNRows=True,
                        inplace=True)

                elif (label_col_type in _INT_TYPES) and __first_train__:
                    self.params.data.label._int_var = self.params.data.label.var

                if __train__ and (label_col_type.startswith('decimal') or (label_col_type in _NUM_TYPES)) \
                        and isinstance(self, RegrEvalMixIn):
                    assert self.params.data.label.excl_outliers \
                       and self.params.data.label.outlier_tails \
                       and self.params.data.label.outlier_tail_proportion \
                       and (self.params.data.label.outlier_tail_proportion < .5)

                    if __first_train__:
                        lower_numeric_null, upper_numeric_null = \
                            self.params.data.nulls.get(
                                self.params.data.label.var,
                                (None, None))

                        self.params.data.label.outlier_tails = \
                            self.params.data.label.outlier_tails.lower()

                        _calc_lower_outlier_threshold = \
                            (self.params.data.label.outlier_tails in ('both', 'lower')) and \
                            pandas.isnull(self.params.data.label.lower_outlier_threshold)

                        _calc_upper_outlier_threshold = \
                            (self.params.data.label.outlier_tails in ('both', 'upper')) and \
                            pandas.isnull(self.params.data.label.upper_outlier_threshold)

                        if _calc_lower_outlier_threshold:
                            if _calc_upper_outlier_threshold:
                                self.params.data.label.lower_outlier_threshold, \
                                self.params.data.label.upper_outlier_threshold = \
                                    adf.quantile(
                                        self.params.data.label.var,
                                        q=(self.params.data.label.outlier_tail_proportion,
                                           1 - self.params.data.label.outlier_tail_proportion),
                                        relativeError=self.params.data.label.outlier_tail_proportion / 3)

                                if (lower_numeric_null is not None) and \
                                        (lower_numeric_null > self.params.data.label.lower_outlier_threshold):
                                    assert lower_numeric_null < self.params.data.label.upper_outlier_threshold, \
                                        '*** {} >= {} ***'.format(
                                            lower_numeric_null,
                                            self.params.data.label.upper_outlier_threshold)

                                    self.params.data.label.lower_outlier_threshold = lower_numeric_null

                                if (upper_numeric_null is not None) and \
                                        (upper_numeric_null < self.params.data.label.upper_outlier_threshold):
                                    assert upper_numeric_null > self.params.data.label.lower_outlier_threshold, \
                                        '*** {} <= {} ***'.format(
                                            upper_numeric_null,
                                            self.params.data.label.lower_outlier_threshold)

                                    self.params.data.label.upper_outlier_threshold = upper_numeric_null

                            else:
                                self.params.data.label.lower_outlier_threshold, \
                                M = adf.quantile(
                                        self.params.data.label.var,
                                        q=(self.params.data.label.outlier_tail_proportion,
                                           1),
                                        relativeError=self.params.data.label.outlier_tail_proportion / 3)

                                if (lower_numeric_null is not None) and \
                                        (lower_numeric_null > self.params.data.label.lower_outlier_threshold):
                                    assert lower_numeric_null < M, \
                                        '*** {} >= {} ***'.format(lower_numeric_null, M)

                                    self.params.data.label.lower_outlier_threshold = lower_numeric_null

                        elif _calc_upper_outlier_threshold:
                            m, self.params.data.label.upper_outlier_threshold = \
                                adf.quantile(
                                    self.params.data.label.var,
                                    q=(0,
                                       1 - self.params.data.label.outlier_tail_proportion),
                                    relativeError=self.params.data.label.outlier_tail_proportion / 3)

                            if (upper_numeric_null is not None) and \
                                    (upper_numeric_null < self.params.data.label.upper_outlier_threshold):
                                assert upper_numeric_null > m, \
                                    '*** {} <= {} ***'.format(upper_numeric_null, m)

                                self.params.data.label.upper_outlier_threshold = upper_numeric_null

                    _lower_outlier_threshold_applicable = \
                        pandas.notnull(self.params.data.label.lower_outlier_threshold)

                    _upper_outlier_threshold_applicable = \
                        pandas.notnull(self.params.data.label.upper_outlier_threshold)

                    if _lower_outlier_threshold_applicable or _upper_outlier_threshold_applicable:
                        _outlier_robust_condition = \
                            ('({0} > {1}) AND ({0} < {2})'.format(
                                self.params.data.label.var,
                                self.params.data.label.lower_outlier_threshold,
                                self.params.data.label.upper_outlier_threshold)
                             if _upper_outlier_threshold_applicable
                             else '{} > {}'.format(
                                self.params.data.label.var,
                                self.params.data.label.lower_outlier_threshold)) \
                            if _lower_outlier_threshold_applicable \
                            else '{} < {}'.format(
                                self.params.data.label.var,
                                self.params.data.label.upper_outlier_threshold)

                        if arimo.debug.ON:
                            self.stdout_logger.debug(
                                msg='*** DATA PREP FOR TRAIN: CONDITION ROBUST TO LABEL OUTLIERS: {} {}... ***\n'
                                    .format(self.params.data.label.var, _outlier_robust_condition))

                        adf('IF({0}, {1}, NULL) AS {1}'.format(
                                _outlier_robust_condition,
                                self.params.data.label.var),
                            *(col for col in adf.columns
                              if col != self.params.data.label.var),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

                adf.alias += self._LABELED_ADF_ALIAS_SUFFIX

        else:
            if isinstance(adf, SparkADF):
                adf_uuid = clean_uuid(uuid.uuid4())

                adf.alias = \
                    '{}__{}__{}'.format(
                        self.params._uuid,
                        __mode__,
                        adf_uuid)

            if __eval__:
                assert self._INT_LABEL_COL not in adf.columns

                label_col_type = adf.type(self.params.data.label.var)

                if isinstance(adf, ArrowADF):
                    if is_float(label_col_type) and isinstance(self, ClassifEvalMixIn):
                        adf.map(
                            mapper=_ArrowADF__castType__pandasDFTransform(
                                    col=self.params.data.label.var,
                                    asType=str,
                                    asCol=None),
                            inheritNRows=True,
                            inplace=True)

                        label_col_type = string()

                    if is_boolean(label_col_type):
                        adf.map(
                            mapper=_ArrowADF__castType__pandasDFTransform(
                                    col=self.params.data.label.var,
                                    asType=int,
                                    asCol=self.params.data.label._int_var),
                            inheritNRows=True,
                            inplace=True)

                    elif is_string(label_col_type):
                        adf.map(
                            mapper=_ArrowADF__encodeStr__pandasDFTransform(
                                    col=self.params.data.label.var,
                                    strs=self.params.data.label._strings,
                                    asCol=self.params.data.label._int_var),
                            inheritNRows=True,
                            inplace=True)

                else:
                    if (label_col_type.startswith('decimal') or (label_col_type in _FLOAT_TYPES)) \
                            and isinstance(self, ClassifEvalMixIn):
                        adf('STRING({0}) AS {0}'.format(self.params.data.label.var),
                            *(col for col in adf.columns
                              if col != self.params.data.label.var),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

                        label_col_type = _STR_TYPE

                    if label_col_type == _BOOL_TYPE:
                        adf('*',
                            'INT({}) AS {}'.format(
                                self.params.data.label.var,
                                self.params.data.label._int_var),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

                    elif label_col_type == _STR_TYPE:
                        adf('*',
                            '(CASE {} ELSE NULL END) AS {}'.format(
                                ' '.join(
                                    "WHEN {} = '{}' THEN {}".format(
                                        self.params.data.label.var, label, i)
                                    for i, label in enumerate(self.params.data.label._strings)),
                                self.params.data.label._int_var),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

                    adf.alias += self._LABELED_ADF_ALIAS_SUFFIX

        if not __from_ppp__:
            # Prepare data into model-able vectors
            if __train__:
                if __first_train__:
                    adf._reprSampleSize = self.params.data.repr_sample_size

                    self.params.data.pred_vars = \
                        set(to_iterable(self.params.data.pred_vars)
                            if self.params.data.pred_vars
                            else adf.possibleFeatureCols) \
                        .union(
                            to_iterable(self.params.data.pred_vars_incl)
                            if self.params.data.pred_vars_incl
                            else []) \
                        .difference(
                            (to_iterable(self.params.data.pred_vars_excl, iterable_type=list)
                             if self.params.data.pred_vars_excl
                             else []) +
                            [self.params.data.label.var,
                             self.params.data.label._int_var])

                    data_transforms_load_path = None
                    data_transforms_save_path = self.data_transforms_dir

                else:
                    data_transforms_load_path = self.data_transforms_dir
                    data_transforms_save_path = None

                adf.maxNCats = self.params.data.max_n_cats

            else:
                data_transforms_load_path = self.data_transforms_dir
                data_transforms_save_path = None

            adf, cat_orig_to_prep_col_map, num_orig_to_prep_col_map = \
                adf.prep(
                    *self.params.data.pred_vars,

                    nulls=self.params.data.nulls,

                    forceCat=self.params.data.force_cat,
                    forceCatIncl=self.params.data.force_cat_incl,
                    forceCatExcl=self.params.data.force_cat_excl,

                    oheCat=__ohe_cat__,
                    scaleCat=__scale_cat__,

                    forceNum=self.params.data.force_num,
                    forceNumIncl=self.params.data.force_num_incl,
                    forceNumExcl=self.params.data.force_num_excl,

                    fill=dict(
                        method=self.params.data.num_null_fill_method,
                        value=None,
                        outlierTails=self.params.data.num_outlier_tails,
                        fillOutliers=False),

                    scaler=self.params.data.num_data_scaler,

                    assembleVec=self.params.data._prep_vec_col
                        if __vectorize__
                        else None,

                    loadPath=data_transforms_load_path,
                    savePath=data_transforms_save_path,

                    returnOrigToPrepColMaps=True,

                    inplace=False,

                    verbose=verbose,

                    alias='{}__{}__{}{}'.format(
                            self.params._uuid,
                            __mode__,
                            adf_uuid,
                            self._PREP_ADF_ALIAS_SUFFIX)
                        if isinstance(adf, SparkADF)
                        else None)

            if __train__ or __eval__:
                if __first_train__:
                    self.params.data.pred_vars = \
                        tuple(sorted(self.params.data.pred_vars
                                     .intersection(set(cat_orig_to_prep_col_map)
                                                   .union(num_orig_to_prep_col_map))))

                    self.params.data.pred_vars_incl = \
                        self.params.data.pred_vars_excl = None

                    self.params.data._cat_prep_cols_metadata = \
                        dict(cat_prep_col_n_metadata
                             for cat_orig_col, cat_prep_col_n_metadata in cat_orig_to_prep_col_map.items()
                             if cat_orig_col in self.params.data.pred_vars)

                    self.params.data._cat_prep_cols = \
                        tuple(sorted(self.params.data._cat_prep_cols_metadata))

                    self.params.data._num_prep_cols_metadata = \
                        dict(num_prep_col_n_metadata
                             for num_orig_col, num_prep_col_n_metadata in num_orig_to_prep_col_map.items()
                             if num_orig_col in self.params.data.pred_vars)

                    self.params.data._num_prep_cols = \
                        tuple(sorted(self.params.data._num_prep_cols_metadata))

                    self.params.data._prep_vec_size = \
                        adf._colWidth(self.params.data._prep_vec_col) \
                        if __vectorize__ \
                        else (len(self.params.data._num_prep_cols) +
                              (sum(_cat_prep_col_metadata['NCats']
                                   for _cat_prep_col_metadata in self.params.data._cat_prep_cols_metadata.values())
                               if cat_orig_to_prep_col_map['__OHE__']
                               else len(self.params.data._cat_prep_cols)))

                if isinstance(adf, ArrowADF):
                    adf = adf[
                        [self.params.data.label.var
                         if self.params.data.label._int_var is None
                         else self.params.data.label._int_var] +
                        ([] if self.params.data.id_col in adf.indexCols
                            else [self.params.data.id_col]) +
                        list(adf.indexCols + adf.tAuxCols +
                             self.params.data._cat_prep_cols + self.params.data._num_prep_cols)]

                elif isinstance(adf, ArrowSparkADF):
                    if __vectorize__:
                        adf(self.params.data.label.var
                                if self.params.data.label._int_var is None
                                else self.params.data.label._int_var,
                            self.params.data._prep_vec_col,
                            *((() if self.params.data.id_col in adf.indexCols
                                  else (self.params.data.id_col,)) +
                              adf.indexCols +
                              adf.tAuxCols),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

                    else:
                        _adf_alias = adf.alias

                        adf = adf[
                            [self.params.data.label.var
                                if self.params.data.label._int_var is None
                                else self.params.data.label._int_var] +
                            ([] if self.params.data.id_col in adf.indexCols
                                else [self.params.data.id_col]) +
                            list(tuple(set(adf.indexCols) - {adf._PARTITION_ID_COL}) +
                                 adf.tAuxCols +
                                 self.params.data._cat_prep_cols + self.params.data._num_prep_cols)]

                        adf.alias = _adf_alias

                else:
                    adf(self.params.data.label.var
                        if self.params.data.label._int_var is None
                        else self.params.data.label._int_var,
                        *((() if self.params.data.id_col in adf.indexCols
                           else (self.params.data.id_col,)) +
                          adf.indexCols +
                          adf.tAuxCols +
                          ((self.params.data._prep_vec_col,)
                           if __vectorize__
                           else (self.params.data._cat_prep_cols + self.params.data._num_prep_cols))),
                        inheritCache=True,
                        inheritNRows=True,
                        inplace=True)

        return adf

    def model(self, ver='latest'):
        return self.__BlueprintedModelClass__(
            blueprint=self,
            ver=ver)

    def model_train_history(self, ver='latest'):
        model = self.model(ver=ver)

        return joblib.load(
            os.path.join(
                model.dir,
                self.params.model._persist.train_history_file))


@_docstr_blueprint
class _DLSupervisedBlueprintABC(_SupervisedBlueprintABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_PARAMS = \
        copy.deepcopy(
            _SupervisedBlueprintABC._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        model=Namespace(
            train=Namespace(
                n_samples_max_multiple_of_data_size=3 ** 3,
                n_train_samples_per_epoch=1000000,
                min_n_val_samples_per_epoch=100000,   # 'all'
                batch_size=500,
                val_batch_size=10000,

                val_metric=Namespace(
                    name='val_loss',
                    mode='min',
                    significance=1e-6),

                reduce_lr_on_plateau=Namespace(
                    patience_n_epochs=9,
                    factor=.3),

                early_stop=Namespace(
                    patience_min_n_epochs=27,
                    patience_proportion_total_n_epochs=.32)),

            _persist=Namespace(
                file='ModelData.h5',
                struct_file='ModelStruct.json',
                weights_file='ModelWeights.h5',
                train_history_file='ModelTrainHistory.pkl')),

        __metadata__={
            'model': Namespace(
                label='Model Params'),

            'model.train': Namespace(
                label='Model-Training Params'),

            'model.train.n_samples_max_multiple_of_data_size': Namespace(
                label='No. of Samples Upper Limit as Multiple of Data Set Size',
                description='No. of Samples Upper Limit as Multiple of Data Set Size',
                type='int',
                default=9),

            'model.train.n_train_samples_per_epoch': Namespace(
                label='No. of Train Samples per Epoch',
                description='No. of Data Samples to Use for Training per Epoch',
                type='int',
                default=1000000),

            'model.train.min_n_val_samples_per_epoch': Namespace(
                label='Min No. of Validation Samples per Epoch',
                description='Min. No. of Data Samples to Representatively Validate Model during each Training Epoch',
                type='int',
                default=100000),

            'model.train.batch_size': Namespace(
                label='Mini-Batch Size',
                description='Size of Each Mini-Batch of Samples Used during Training',
                type='int',
                default=500),

            'model.train.val_batch_size': Namespace(
                label='Batch Size for Validation',
                description='No. of Data Cases to Validate Together at One Time',
                type='int',
                default=10000),

            'model.train.val_metric': Namespace(),

            'model.train.val_metric.name': Namespace(),

            'model.train.val_metric.mode': Namespace(),

            'model.train.val_metric.significance': Namespace(),

            'model.train.reduce_lr_on_plateau': Namespace(),

            'model.train.reduce_lr_on_plateau.patience_n_epochs': Namespace(),

            'model.train.reduce_lr_on_plateau.factor': Namespace(),

            'model.train.early_stop': Namespace(),

            'model.train.early_stop.patience_min_n_epochs': Namespace(),

            'model.train.early_stop.patience_proportion_total_n_epochs': Namespace(),

            'model._persist': Namespace(),

            'model._persist.file': Namespace(),

            'model._persist.struct_file': Namespace(),

            'model._persist.weights_file': Namespace(),

            'model._persist.train_history_file': Namespace()})

    DEFAULT_MODEL_TRAIN_MAX_GEN_QUEUE_SIZE = 10 ** 3   # sufficient to keep CPUs busy while feeding into GPU
    DEFAULT_MODEL_TRAIN_N_WORKERS = 9   # not too many in order to avoid OOM
    DEFAULT_MODEL_TRAIN_N_GPUS = 1

    def _derive_model_train_params(self, data_size=None):
        # derive _n_samples
        self.params.model.train._n_samples = \
            min(self.params.model.train.n_samples,
                self.params.model.train.n_samples_max_multiple_of_data_size * data_size) \
            if self.params.model.train.n_samples_max_multiple_of_data_size and data_size \
            else self.params.model.train.n_samples

        # derive train_proportion, _n_train_samples & _n_val_samples
        if self.params.model.train.val_proportion:
            self.params.model.train.train_proportion = \
                1 - self.params.model.train.val_proportion

            self.params.model.train._n_train_samples = \
                int(round(self.params.model.train.train_proportion *
                          self.params.model.train._n_samples))

            self.params.model.train._n_val_samples = \
                self.params.model.train._n_samples - \
                self.params.model.train._n_train_samples

            self.params.model.train.n_cross_val_folds = None

        else:
            self.params.model.train.train_proportion = 1

            self.params.model.train._n_train_samples = \
                self.params.model.train._n_samples

            self.params.model.train._n_val_samples = 0

        # derive _n_train_samples_per_epoch & _n_epochs
        if self.params.model.train.n_train_samples_per_epoch <= self.params.model.train._n_train_samples:
            self.params.model.train._n_train_samples_per_epoch = \
                self.params.model.train.n_train_samples_per_epoch

            self.params.model.train._n_epochs = \
                int(math.ceil(self.params.model.train._n_train_samples /
                              self.params.model.train._n_train_samples_per_epoch))

        else:
            self.params.model.train._n_train_samples_per_epoch = \
                self.params.model.train._n_train_samples

            self.params.model.train._n_epochs = 1

        # derive _n_val_samples_per_era & _n_val_samples_per_epoch
        if self.params.model.train.min_n_val_samples_per_epoch == 'all':
            self.params.model.train._n_val_samples_per_epoch = \
                self.params.model.train._n_val_samples

        else:
            self.params.model.train._n_val_samples_per_epoch = \
                max(self.params.model.train.min_n_val_samples_per_epoch,
                    int(math.ceil(self.params.model.train._n_val_samples /
                                  self.params.model.train._n_epochs)))

        # derive _n_train_batches_per_epoch & _n_val_batches_per_epoch
        self.params.model.train._n_train_batches_per_epoch = \
            int(math.ceil(self.params.model.train._n_train_samples_per_epoch /
                          self.params.model.train.batch_size))

        self.params.model.train._n_val_batches_per_epoch = \
            int(math.ceil(self.params.model.train._n_val_samples_per_epoch /
                          self.params.model.train.val_batch_size))


@_docstr_blueprint
class _PPPBlueprintABC(_BlueprintABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_PARAMS = \
        copy.deepcopy(
            _BlueprintABC._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        model=Namespace(
            component_blueprints={}),

        __metadata__={
            'model': Namespace(
                label='Model Params'),

            'model.component_blueprints': Namespace()})

    _TO_SCORE_ALL_VARS_ADF_ALIAS_SUFFIX = '__toScoreAllVars'

    GOOD_COMPONENT_BLUEPRINT_MIN_R2 = .68
    GOOD_COMPONENT_BLUEPRINT_MAX_MAE_MedAE_RATIO = 3

    _SGN_PREFIX = 'sgn__'
    _ABS_PREFIX = 'abs__'
    _NEG_PREFIX = 'neg__'
    _POS_PREFIX = 'pos__'
    _SGN_PREFIXES = _SGN_PREFIX, _ABS_PREFIX, _NEG_PREFIX, _POS_PREFIX

    _BENCHMARK_METRICS_ADF_ALIAS = '__BenchmarkMetrics__'

    _GLOBAL_PREFIX = 'global__'
    _INDIV_PREFIX = 'indiv__'
    _GLOBAL_OR_INDIV_PREFIX = ''
    _GLOBAL_OR_INDIV_PREFIXES = _GLOBAL_OR_INDIV_PREFIX,

    _RAW_METRICS = 'MAE',   # 'MedAE', 'RMSE'

    _ERR_MULT_COLS = \
        dict(MedAE='MedAE_Mult',
             MAE='MAE_Mult',
             RMSE='RMSE_Mult')

    _ERR_MULT_PREFIXES = \
        {k: (v + '__')
         for k, v in _ERR_MULT_COLS.items()}

    _rowEuclNorm_PREFIX = 'rowEuclNorm__'
    _rowSumOfLog_PREFIX = 'rowSumOfLog__'
    _rowHigh_PREFIX = 'rowHigh__'
    _rowLow_PREFIX = 'rowLow__'
    _rowMean_PREFIX = 'rowMean__'
    _rowGMean_PREFIX = 'rowGMean__'
    _ROW_SUMM_PREFIXES = \
        _rowEuclNorm_PREFIX, \
        _rowSumOfLog_PREFIX, \
        _rowHigh_PREFIX, \
        _rowLow_PREFIX, \
        _rowMean_PREFIX, \
        _rowGMean_PREFIX

    _ROW_ERR_MULT_SUMM_COLS = \
        [(_row_summ_prefix + _ABS_PREFIX + _global_or_indiv_prefix + _ERR_MULT_COLS[_metric])
         for _metric, _global_or_indiv_prefix, _row_summ_prefix in
         itertools.product(_RAW_METRICS, _GLOBAL_OR_INDIV_PREFIXES, _ROW_SUMM_PREFIXES)]

    _dailyMed_PREFIX = 'dailyMed__'
    _dailyMean_PREFIX = 'dailyMean__'
    _dailyMax_PREFIX = 'dailyMax__'
    _dailyMin_PREFIX = 'dailyMin__'
    _DAILY_SUMM_PREFIXES = _dailyMed_PREFIX, _dailyMean_PREFIX, _dailyMax_PREFIX, _dailyMin_PREFIX

    _DAILY_ERR_MULT_SUMM_COLS = \
        [(_daily_summ_prefix + _row_err_mult_summ_col)
         for _row_err_mult_summ_col, _daily_summ_prefix in
         itertools.product(_ROW_ERR_MULT_SUMM_COLS, _DAILY_SUMM_PREFIXES)]

    _EWMA_PREFIX = 'ewma'

    def __init__(self, *args, **kwargs):
        model_params = kwargs.pop('__model_params__', {})

        super(_PPPBlueprintABC, self).__init__(*args, **kwargs)

        assert self.params.data.id_col and self.params.data.time_col

        for label_var_name, component in self.params.model.component_blueprints.items():
            if isinstance(component, _BlueprintABC):
                assert isinstance(component, _SupervisedBlueprintABC), \
                    'All Component Blueprints Must Be Supervised'

                self.params.model.component_blueprints[label_var_name] = \
                    copy.deepcopy(component.params)

            else:
                assert isinstance(component, (dict, Namespace))

            self.params.model.component_blueprints[label_var_name].data.id_col = self.params.data.id_col
            self.params.model.component_blueprints[label_var_name].data.time_col = self.params.data.time_col

            self.params.model.component_blueprints[label_var_name].data.nulls = self.params.data.nulls

            self.params.model.component_blueprints[label_var_name].data.label.var = label_var_name

            self.params.model.component_blueprints[label_var_name].model.update(model_params)

            self.params.model.component_blueprints[label_var_name].persist = self.params.persist

    def __repr__(self):
        return '{} Instance "{}" (Monitored: {})'.format(
            self.__qual_name__(),
            self.params.uuid,
            ', '.join('"{}"'.format(label_var_name)
                      for label_var_name in self.params.model.component_blueprints))

    def prep_data(self, df, __mode__='score', __vectorize__=True, verbose=True, **kwargs):
        # check if (incrementally-)training, scoring or evaluating
        if __mode__ == self._TRAIN_MODE:
            __train__ = True
            __score__ = __eval__ = False

        elif __mode__ == self._SCORE_MODE:
            __train__ = __eval__ = False
            __score__ = True

        elif __mode__ == self._EVAL_MODE:
            __train__ = __score__ = False
            __eval__ = True

        else:
            raise ValueError(
                '*** Blueprint "{}" argument must be one of "{}", "{}" and "{}" ***'
                    .format(self._MODE_ARG, self._TRAIN_MODE, self._SCORE_MODE, self._EVAL_MODE))

        assert __train__ + __score__ + __eval__ == 1

        __first_train__ = __train__ and (not os.path.isdir(self.data_transforms_dir))

        if isinstance(df, SparkADF):
            adf = df

            adf.tCol = self.params.data.time_col

            if isinstance(self, _TimeSerDataPrepMixInABC):
                adf.iCol = self.params.data.id_col

        else:
            kwargs['tCol'] = self.params.data.time_col

            if isinstance(self, _TimeSerDataPrepMixInABC):
                kwargs['iCol'] = self.params.data.id_col

            if isinstance(df, pandas.DataFrame):
                adf = SparkADF.create(
                        data=df,
                        schema=None,
                        samplingRatio=None,
                        verifySchema=False,
                        **kwargs)

            elif isinstance(df, pyspark.sql.DataFrame):
                adf = SparkADF(sparkDF=df, **kwargs)

            elif isinstance(df, ArrowADF):
                adf = df

                adf.tCol = self.params.data.time_col

                if isinstance(self, _TimeSerDataPrepMixInABC):
                    adf.iCol = self.params.data.id_col

            else:
                assert isinstance(df, _STR_CLASSES)

                adf = (ArrowSparkADF
                       if arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS
                       else ArrowADF)(
                    path=df, **kwargs)

        assert (self.params.data.id_col in adf.columns) \
           and (self.params.data.time_col in adf.columns)

        if __train__:
            if isinstance(adf, SparkADF):
                if isinstance(adf, ArrowSparkADF):
                    __vectorize__ = False

                adf_uuid = clean_uuid(uuid.uuid4())

                adf.alias = \
                    '{}__{}__{}'.format(
                        self.params._uuid,
                        __mode__,
                        adf_uuid)

            if __first_train__:
                adf._reprSampleSize = self.params.data.repr_sample_size

                cols_to_prep = set()

                for label_var_name in set(self.params.model.component_blueprints).intersection(adf.contentCols):
                    component_blueprint_params = \
                        self.params.model.component_blueprints[label_var_name]

                    component_blueprint_params.data.pred_vars = \
                        set(to_iterable(component_blueprint_params.data.pred_vars)
                            if component_blueprint_params.data.pred_vars
                            else adf.possibleFeatureCols) \
                        .union(
                            to_iterable(component_blueprint_params.data.pred_vars_incl)
                            if component_blueprint_params.data.pred_vars_incl
                            else []) \
                        .difference(
                            (to_iterable(component_blueprint_params.data.pred_vars_excl, iterable_type=list)
                             if component_blueprint_params.data.pred_vars_excl
                             else []) +
                            [label_var_name])

                    cols_to_prep.update(component_blueprint_params.data.pred_vars)

                    component_blueprint_params.data.pred_vars_incl = \
                        component_blueprint_params.data.pred_vars_excl = None

                data_transforms_load_path = None
                data_transforms_save_path = self.data_transforms_dir

            else:
                data_transforms_load_path = self.data_transforms_dir
                data_transforms_save_path = None

                cols_to_prep = ()

            adf.maxNCats = self.params.data.max_n_cats

        else:
            if isinstance(adf, SparkADF):
                adf_uuid = clean_uuid(uuid.uuid4())

                adf.filter(
                    condition="({0} IS NOT NULL) AND (STRING({0}) != 'NaN') AND ({1} IS NOT NULL) AND (STRING({1}) != 'NaN')"
                        .format(self.params.data.id_col, self.params.data.time_col),
                    alias='{}__{}__{}'.format(
                        self.params._uuid,
                        __mode__,
                        adf_uuid),
                    inplace=True)

            data_transforms_load_path = self.data_transforms_dir
            data_transforms_save_path = None

            cols_to_prep = ()

            orig_cols_to_keep = set(adf.indexCols)
            orig_cols_to_keep.add(self.params.data.id_col)

        adf, cat_orig_to_prep_col_map, num_orig_to_prep_col_map = \
            adf.prep(
                *cols_to_prep,

                nulls=self.params.data.nulls,

                forceCat=self.params.data.force_cat,
                forceCatIncl=self.params.data.force_cat_incl,
                forceCatExcl=self.params.data.force_cat_excl,

                oheCat=False,
                scaleCat=True,

                forceNum=self.params.data.force_num,
                forceNumIncl=self.params.data.force_num_incl,
                forceNumExcl=self.params.data.force_num_excl,

                fill=dict(
                    method=self.params.data.num_null_fill_method,
                    value=None,
                    outlierTails=self.params.data.num_outlier_tails,
                    fillOutliers=False),

                scaler=self.params.data.num_data_scaler,

                assembleVec=None,

                loadPath=data_transforms_load_path,
                savePath=data_transforms_save_path,

                returnOrigToPrepColMaps=True,

                inplace=False,

                verbose=verbose)

        if __train__:
            if isinstance(adf, SparkADF):
                adf.alias = \
                    '{}__{}__{}{}'.format(
                        self.params._uuid,
                        __mode__,
                        adf_uuid,
                        self._PREP_ADF_ALIAS_SUFFIX)

            component_labeled_adfs = Namespace()

            for label_var_name in set(self.params.model.component_blueprints).intersection(adf.contentCols):
                if adf.suffNonNull(label_var_name):
                    component_blueprint_params = \
                        self.params.model.component_blueprints[label_var_name]

                    if __first_train__:
                        component_blueprint_params.data.pred_vars = \
                            tuple(sorted(component_blueprint_params.data.pred_vars
                                         .intersection(set(cat_orig_to_prep_col_map)
                                                       .union(num_orig_to_prep_col_map))))

                        component_blueprint_params.data.pred_vars_incl = \
                            component_blueprint_params.data.pred_vars_excl = None

                        component_blueprint_params.data._cat_prep_cols_metadata = \
                            dict(cat_prep_col_n_metadata
                                 for cat_orig_col, cat_prep_col_n_metadata in cat_orig_to_prep_col_map.items()
                                 if cat_orig_col in component_blueprint_params.data.pred_vars)

                        component_blueprint_params.data._cat_prep_cols = \
                            tuple(sorted(component_blueprint_params.data._cat_prep_cols_metadata))

                        component_blueprint_params.data._num_prep_cols_metadata = \
                            dict(num_prep_col_n_metadata
                                 for num_orig_col, num_prep_col_n_metadata in num_orig_to_prep_col_map.items()
                                 if num_orig_col in component_blueprint_params.data.pred_vars)

                        component_blueprint_params.data._num_prep_cols = \
                            tuple(sorted(component_blueprint_params.data._num_prep_cols_metadata))

                        component_blueprint_params.data._prep_vec_size = \
                            len(component_blueprint_params.data._num_prep_cols) + \
                            (sum(_cat_prep_col_metadata['NCats']
                                 for _cat_prep_col_metadata in component_blueprint_params.data._cat_prep_cols_metadata.values())
                             if cat_orig_to_prep_col_map['__OHE__']
                             else len(component_blueprint_params.data._cat_prep_cols))

                    if isinstance(adf, SparkADF):
                        if (__vectorize__ is None) or __vectorize__:
                            component_labeled_adfs[label_var_name] = \
                                adf(VectorAssembler(
                                        inputCols=component_blueprint_params.data._cat_prep_cols +
                                                  component_blueprint_params.data._num_prep_cols,
                                        outputCol=component_blueprint_params.data._prep_vec_col).transform,
                                    inheritCache=True,
                                    inheritNRows=True)(
                                    label_var_name,
                                    component_blueprint_params.data._prep_vec_col,
                                    *((() if self.params.data.id_col in adf.indexCols
                                          else (self.params.data.id_col,)) +
                                      adf.indexCols + adf.tAuxCols),
                                    alias=adf.alias + '__LABEL__' + label_var_name,
                                    inheritCache=True,
                                    inheritNRows=True)

                        else:
                            _adf_alias = adf.alias

                            component_labeled_adf = \
                                adf[[label_var_name] +
                                    list((() if self.params.data.id_col in adf.indexCols
                                             else (self.params.data.id_col,)) +
                                         (tuple(set(adf.indexCols) - {adf._PARTITION_ID_COL})
                                          if isinstance(adf, ArrowSparkADF)
                                          else adf.indexCols) +
                                         adf.tAuxCols +
                                         component_blueprint_params.data._cat_prep_cols +
                                         component_blueprint_params.data._num_prep_cols)]

                            component_labeled_adf.alias = \
                                _adf_alias + '__LABEL__' + label_var_name

                            component_labeled_adfs[label_var_name] = component_labeled_adf

                    else:
                        component_labeled_adfs[label_var_name] = \
                            adf[[label_var_name] +
                                list((() if self.params.data.id_col in adf.indexCols
                                         else (self.params.data.id_col,)) +
                                     adf.indexCols +   # adf.tAuxCols +
                                     component_blueprint_params.data._cat_prep_cols +
                                     component_blueprint_params.data._num_prep_cols)]

            # save Blueprint & data transforms
            if __first_train__:
                self.save()

            return component_labeled_adfs

        elif isinstance(adf, SparkADF):
            adf.alias = \
                '{}__{}__{}{}'.format(
                    self.params._uuid,
                    __mode__,
                    adf_uuid,
                    self._PREP_ADF_ALIAS_SUFFIX)

            if __vectorize__:
                vector_assemblers = []
                vector_cols = []

                for label_var_name, component_blueprint_params in self.params.model.component_blueprints.items():
                    if (label_var_name in adf.columns) and component_blueprint_params.model.ver:
                        orig_cols_to_keep.add(label_var_name)

                        component_blueprint_params = \
                            self.params.model.component_blueprints[label_var_name]

                        vector_col = component_blueprint_params.data._prep_vec_col + label_var_name

                        vector_assemblers.append(
                            VectorAssembler(
                                inputCols=component_blueprint_params.data._cat_prep_cols +
                                          component_blueprint_params.data._num_prep_cols,
                                outputCol=vector_col))

                        vector_cols.append(vector_col)

                return adf(PipelineModel(stages=vector_assemblers).transform,
                           inheritCache=True,
                           inheritNRows=True)(
                    *(orig_cols_to_keep.union(vector_cols)),
                    alias=adf.alias + self._TO_SCORE_ALL_VARS_ADF_ALIAS_SUFFIX,
                    inheritCache=True,
                    inheritNRows=True)

            else:
                prep_cols = set()

                for label_var_name, component_blueprint_params in self.params.model.component_blueprints.items():
                    if (label_var_name in adf.columns) and component_blueprint_params.model.ver:
                        orig_cols_to_keep.add(label_var_name)

                        prep_cols.update(
                            component_blueprint_params.data._cat_prep_cols +
                            component_blueprint_params.data._num_prep_cols)

                return adf(
                    *(orig_cols_to_keep.union(prep_cols)),
                    alias=adf.alias + self._TO_SCORE_ALL_VARS_ADF_ALIAS_SUFFIX,
                    inheritCache=True,
                    inheritNRows=True)

        else:
            prep_cols = set()

            for label_var_name, component_blueprint_params in self.params.model.component_blueprints.items():
                if (label_var_name in adf.columns) and component_blueprint_params.model.ver:
                    orig_cols_to_keep.add(label_var_name)

                    prep_cols.update(
                        component_blueprint_params.data._cat_prep_cols +
                        component_blueprint_params.data._num_prep_cols)

            return adf[list(orig_cols_to_keep) + sorted(prep_cols)]

    def train(self, *args, **kwargs):
        __gen_queue_size__ = \
            kwargs.pop(
                '__gen_queue_size__',
                _DLSupervisedBlueprintABC.DEFAULT_MODEL_TRAIN_MAX_GEN_QUEUE_SIZE)
        assert __gen_queue_size__, \
            '*** __gen_queue_size__ = {} ***'.format(__gen_queue_size__)

        __n_workers__ = \
            kwargs.pop(
                '__n_workers__',
                _DLSupervisedBlueprintABC.DEFAULT_MODEL_TRAIN_N_WORKERS)
        assert __n_workers__, \
            '*** __n_workers__ = {} ***'.format(__n_workers__)

        __multiproc__ = kwargs.pop('__multiproc__', True)

        __n_gpus__ = \
            kwargs.pop(
                '__n_gpus__',
                _DLSupervisedBlueprintABC.DEFAULT_MODEL_TRAIN_N_GPUS)
        assert __n_gpus__, \
            '*** __n_gpus__ = {} ***'.format(__n_gpus__)

        __cpu_merge__ = bool(kwargs.pop('__cpu_merge__', True))
        __cpu_reloc__ = bool(kwargs.pop('__cpu_reloc__', False))   # *** cpu_relocation MAKES TEMPLATE MODEL WEIGHTS FAIL TO UPDATE ***

        # whether to retrain component Blueprinted models
        __retrain_components__ = kwargs.pop('__retrain_components__', False)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        component_labeled_adfs = \
            self.prep_data(
                __mode__=self._TRAIN_MODE,
                verbose=verbose,
                *args, **kwargs)

        models = Namespace()

        for label_var_name, component_labeled_adf in component_labeled_adfs.items():
            blueprint_params = self.params.model.component_blueprints[label_var_name]

            if blueprint_params.model.ver and __retrain_components__:
                blueprint_params.model.ver = None

                blueprint_params.data.label.lower_outlier_threshold = \
                    blueprint_params.data.label.upper_outlier_threshold = \
                    None

            else:
                blueprint_params.uuid = \
                    '{}---{}---{}'.format(
                        self.params.uuid,
                        label_var_name,
                        uuid.uuid4())

            blueprint = \
                _blueprint_from_params(
                    blueprint_params=blueprint_params,
                    aws_access_key_id=self.auth.aws.access_key_id,
                    aws_secret_access_key=self.auth.aws.secret_access_key,
                    verbose=False)

            if blueprint_params.model.ver is None:
                if blueprint_params.model.train.objective is None:
                    blueprint_params.model.train.objective = \
                        blueprint.params.model.train.objective = \
                            'MAE'   # mae / mean_absolute_error
                            # MAPE / mape / mean_absolute_percentage_error   *** ASSUMES POSITIVE LABELS ***
                            # MSLE / msle / mean_squared_logarithmic_error   *** ASSUMES POSITIVE LABELS ***
                            # *** MSE / mean_squared_error / mse   TOO SENSITIVE TO LARGE OUTLIERS ***

                model = \
                    blueprint.train(
                        df=component_labeled_adf,
                        __from_ppp__=True,
                        __gen_queue_size__=__gen_queue_size__,
                        __multiproc__=__multiproc__,
                        __n_workers__=__n_workers__,
                        __n_gpus__=__n_gpus__,
                        __cpu_merge__=__cpu_merge__,
                        __cpu_reloc__=__cpu_reloc__,
                        verbose=verbose)

                # *** COPY DataTransforms AFTER model training so that __first_train__ is correctly detected ***
                if not os.path.isdir(blueprint.data_transforms_dir):
                    fs.cp(from_path=self.data_transforms_dir,
                          to_path=blueprint.data_transforms_dir,
                          hdfs=False,
                          is_dir=True)

                # re-save blueprint to make sure DataTransforms dir is sync'ed to S3
                blueprint.save()

                blueprint_params.model.ver = \
                    blueprint.params.model.ver = \
                    model.ver

                blueprint_params.data.label = \
                    blueprint.params.data.label

                # update timestamp of component blueprint
                blueprint_params[_TIMESTAMP_ARG_NAME] = \
                    blueprint.params[_TIMESTAMP_ARG_NAME]

                models[label_var_name] = model

            else:
                models[label_var_name] = \
                    blueprint.model(
                        ver=blueprint_params.model.ver)

        # save Blueprint, with updated component blueprint params
        self.save()

        return Namespace(
            blueprint=self,
            models=models)

    def eval(self, *args, **kwargs):
        # whether to exclude outlying labels from eval
        __excl_outliers__ = kwargs.pop('__excl_outliers__', True)

        # whether to save
        save = kwargs.pop('save', False)
        if save:
            assert __excl_outliers__

        # whether to cache data at certain stages
        __cache_vector_data__ = kwargs.pop('__cache_vector_data__', False)
        __cache_tensor_data__ = kwargs.pop('__cache_tensor_data__', False)

        # whether to repartition by ID
        __partition_by_id__ = kwargs.pop('__partition_by_id__', False)
        __coalesce__ = kwargs.pop('__coalesce__', False)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        adf = self.score(
            __mode__=self._EVAL_MODE,
            __cache_vector_data__=__cache_vector_data__,
            __cache_tensor_data__=__cache_tensor_data__,
            verbose=verbose,
            *args, **kwargs)

        label_var_names_n_blueprint_params = \
            {label_var_name: blueprint_params
             for label_var_name, blueprint_params in self.params.model.component_blueprints.items()
             if (label_var_name in adf.columns) and blueprint_params.model.ver}

        # cache to calculate multiple metrics quickly
        if len(label_var_names_n_blueprint_params) > 1:
            adf.cache(
                eager=True,
                verbose=verbose)

        eval_metrics = {}

        id_col = self.params.data.id_col
        id_col_type_is_str = (adf.type(id_col) == _STR_TYPE)

        for label_var_name, blueprint_params in label_var_names_n_blueprint_params.items():
            score_col_name = blueprint_params.model.score.raw_score_col_prefix + label_var_name

            # *** SPARK 2.3.0/1 BUG *** >>>
            # NOT CACHING BEFORE FILTERING RESULTS IN
            # "WARN TaskMemoryManager:302 - Failed to allocate a page (... bytes) try again"
            _per_label_adf_pre_cached = \
                adf[id_col, score_col_name, label_var_name]

            _per_label_adf_pre_cached.cache(
                eager=True,
                verbose=verbose)
            # ^^^ *** SPARK 2.3.0/1 BUG ***

            _per_label_adf = \
                _per_label_adf_pre_cached.filter(
                    condition="({0} IS NOT NULL) AND (STRING({0}) != 'NaN') AND ({1} IS NOT NULL) AND (STRING({1}) != 'NaN')"
                        .format(label_var_name, score_col_name))

            blueprint = \
                _blueprint_from_params(
                    blueprint_params=self.params.model.component_blueprints[label_var_name],
                    aws_access_key_id=self.auth.aws.access_key_id,
                    aws_secret_access_key=self.auth.aws.secret_access_key,
                    verbose=False)

            # whether to exclude outlying labels
            _lower_outlier_threshold_applicable = \
                pandas.notnull(blueprint_params.data.label.lower_outlier_threshold)

            _upper_outlier_threshold_applicable = \
                pandas.notnull(blueprint_params.data.label.upper_outlier_threshold)

            _excl_outliers = __excl_outliers__ and \
                isinstance(blueprint, RegrEvalMixIn) and \
                blueprint_params.data.label.excl_outliers and \
                blueprint_params.data.label.outlier_tails and \
                blueprint_params.data.label.outlier_tail_proportion and \
                (blueprint_params.data.label.outlier_tail_proportion < .5) and \
                (_lower_outlier_threshold_applicable or _upper_outlier_threshold_applicable)

            if _excl_outliers:
                label_var_type = adf.type(label_var_name)
                assert (label_var_type.startswith('decimal') or (label_var_type in _NUM_TYPES))

                _outlier_robust_condition = \
                    ('{} BETWEEN {} AND {}'
                        .format(
                            label_var_name,
                            blueprint_params.data.label.lower_outlier_threshold,
                            blueprint_params.data.label.upper_outlier_threshold)
                     if _upper_outlier_threshold_applicable
                     else '{} > {}'.format(
                            label_var_name,
                            blueprint_params.data.label.lower_outlier_threshold)) \
                    if _lower_outlier_threshold_applicable \
                    else '{} < {}'.format(
                            label_var_name,
                            blueprint_params.data.label.upper_outlier_threshold)

                if arimo.debug.ON:
                    self.stdout_logger.debug(
                        msg='*** EVAL: CONDITION ROBUST TO OUTLIER LABELS: {} {} ***\n'
                            .format(label_var_name, _outlier_robust_condition))

                _per_label_adf.filter(
                    condition=_outlier_robust_condition,
                    inplace=True)

            if __partition_by_id__:
                ids = _per_label_adf._sparkDF[[id_col]].distinct().toPandas()[id_col]

                n_ids = len(ids)

                _per_label_adf = \
                    _per_label_adf.repartition(
                        n_ids,
                        id_col,
                        alias=adf.alias + '__toEval__' + label_var_name)

                # cache to calculate multiple metrics quickly
                _per_label_adf.cache(
                    eager=True,
                    verbose=verbose)

                eval_metrics[label_var_name] = \
                    {self._GLOBAL_EVAL_KEY: dict(n=arimo.eval.metrics.n(_per_label_adf)),
                     self._BY_ID_EVAL_KEY: {}}

                evaluators = []

                for metric_name in blueprint.eval_metrics:
                    metric_class = getattr(arimo.eval.metrics, metric_name)

                    evaluator = metric_class(
                        label_col=label_var_name,
                        score_col=score_col_name)

                    evaluators.append(evaluator)

                    eval_metrics[label_var_name][self._GLOBAL_EVAL_KEY][evaluator.name] = \
                        evaluator(_per_label_adf)

                for i in tqdm.tqdm(range(n_ids)):
                    _per_label_partitioned_by_id_spark_df = \
                        arimo.backend.spark.createDataFrame(
                            data=_per_label_adf.rdd.mapPartitionsWithIndex(
                                f=lambda splitIndex, iterator:
                                    iterator
                                    if splitIndex == i
                                    else [],
                                preservesPartitioning=False),
                            schema=_per_label_adf.schema,
                            samplingRatio=None,
                            verifySchema=False)

                    if __coalesce__:
                        _per_label_partitioned_by_id_spark_df = \
                            _per_label_partitioned_by_id_spark_df.coalesce(numPartitions=1)

                    # cache to calculate multiple metrics quickly
                    _per_label_partitioned_by_id_spark_df.cache()
                    _n_rows = _per_label_partitioned_by_id_spark_df.count()

                    if _n_rows:
                        _ids = _per_label_partitioned_by_id_spark_df[[id_col]].distinct().toPandas()[id_col]

                        if len(_ids) > 1:
                            for _id in _ids:
                                _per_label_per_id_spark_df = \
                                    _per_label_partitioned_by_id_spark_df \
                                        .filter(
                                            condition="{} = {}"
                                                .format(
                                                    id_col,
                                                    "'{}'".format(_id))
                                                        if id_col_type_is_str
                                                        else _id) \
                                        .drop(id_col)

                                _per_label_per_id_spark_df.cache()
                                _per_label_per_id_spark_df.count()

                                eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][_id] = \
                                    dict(n=arimo.eval.metrics.n(_per_label_per_id_spark_df))

                                for evaluator in evaluators:
                                    eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][_id][evaluator.name] = \
                                        evaluator(_per_label_per_id_spark_df)

                                _per_label_per_id_spark_df.unpersist()

                        else:
                            _id = _ids[0]

                            eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][_id] = \
                                dict(n=arimo.eval.metrics.n(_per_label_partitioned_by_id_spark_df))

                            for evaluator in evaluators:
                                eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][_id][evaluator.name] = \
                                    evaluator(_per_label_partitioned_by_id_spark_df)

                    _per_label_partitioned_by_id_spark_df.unpersist()

            else:
                _per_label_adf.alias = \
                    adf.alias + '__toEval__' + label_var_name

                # cache to calculate multiple metrics quickly
                _per_label_adf.cache(
                    eager=True,
                    verbose=verbose)

                eval_metrics[label_var_name] = \
                    {self._GLOBAL_EVAL_KEY: dict(n=arimo.eval.metrics.n(_per_label_adf)),
                     self._BY_ID_EVAL_KEY: {}}

                evaluators = []

                for metric_name in blueprint.eval_metrics:
                    metric_class = getattr(arimo.eval.metrics, metric_name)

                    evaluator = metric_class(
                        label_col=label_var_name,
                        score_col=score_col_name)

                    evaluators.append(evaluator)

                    eval_metrics[label_var_name][self._GLOBAL_EVAL_KEY][evaluator.name] = \
                        evaluator(_per_label_adf)

                ids = _per_label_adf._sparkDF[[id_col]].distinct().toPandas()[id_col]

                for _id in tqdm.tqdm(ids):
                    _per_label_per_id_adf = \
                        _per_label_adf \
                            .filter(
                                condition="{} = {}"
                                    .format(
                                        id_col,
                                        "'{}'".format(_id)
                                            if id_col_type_is_str
                                            else _id)) \
                            .drop(
                                id_col,
                                alias=(_per_label_adf.alias + '__' + clean_str(clean_uuid(id)))
                                    if arimo.debug.ON
                                    else None)

                    # cache to calculate multiple metrics quickly
                    _per_label_per_id_adf.cache(
                        eager=True,
                        verbose=False)

                    eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][_id] = \
                        dict(n=arimo.eval.metrics.n(_per_label_per_id_adf))

                    for evaluator in evaluators:
                        eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][_id][evaluator.name] = \
                            evaluator(_per_label_per_id_adf)

                    _per_label_per_id_adf.unpersist()

            _per_label_adf.unpersist()

            # *** SPARK 2.3.0/1 BUG *** >>>
            _per_label_adf_pre_cached.unpersist()
            # ^^^ *** SPARK 2.3.0/1 BUG ***

        adf.unpersist()

        if save:
            self.params.benchmark_metrics = eval_metrics
            self.save()

        return eval_metrics

    @classmethod
    def _is_good_component_blueprint(cls, label_var_name, benchmark_metrics_for_label_var_name=None, blueprint_obj=None):
        if not benchmark_metrics_for_label_var_name:
            benchmark_metrics_for_label_var_name = \
                blueprint_obj.params.benchmark_metrics.get(label_var_name)

        if benchmark_metrics_for_label_var_name:
            global_benchmark_metrics_for_label_var_name = \
                benchmark_metrics_for_label_var_name[cls._GLOBAL_EVAL_KEY]

            r2 = global_benchmark_metrics_for_label_var_name['R2']

            if r2 > cls.GOOD_COMPONENT_BLUEPRINT_MIN_R2:
                mae = global_benchmark_metrics_for_label_var_name['MAE']
                medae = global_benchmark_metrics_for_label_var_name['MedAE']
                mae_medae_ratio = mae / medae

                if mae_medae_ratio < cls.GOOD_COMPONENT_BLUEPRINT_MAX_MAE_MedAE_RATIO:
                    return True

                else:
                    msg = '*** {}: {}: MAE / MedAE = {:.3g} / {:.3g} = {:.3g}; R2 = {:.3f} ***'.format(
                        blueprint_obj, label_var_name,
                        mae, medae, mae_medae_ratio, r2)

                    assert r2 > .9, msg

                    print(msg)

                    return False

            else:
                print('*** {}: {}: R2 = {:.3f} ***'.format(blueprint_obj, label_var_name, r2))
                return False

    def err_mults(self, df, *label_var_names):
        score_col_names = {}
        lower_outlier_thresholds = {}
        upper_outlier_thresholds = {}

        if label_var_names:
            _label_var_names = []

            for label_var_name in set(label_var_names).intersection(self.params.model.component_blueprints).intersection(df.columns):
                component_blueprint_params = self.params.model.component_blueprints[label_var_name]

                if component_blueprint_params.model.ver and \
                        self._is_good_component_blueprint(
                            label_var_name=label_var_name,
                            blueprint_obj=self):
                    _label_var_names.append(label_var_name)

                    score_col_names[label_var_name] = \
                        component_blueprint_params.model.score.raw_score_col_prefix + label_var_name

                    lower_outlier_thresholds[label_var_name] = \
                        component_blueprint_params.data.label.lower_outlier_threshold

                    upper_outlier_thresholds[label_var_name] = \
                        component_blueprint_params.data.label.upper_outlier_threshold

            label_var_names = _label_var_names

        else:
            label_var_names = []

            for label_var_name, component_blueprint_params in self.params.model.component_blueprints.items():
                if (label_var_name in df.columns) and component_blueprint_params.model.ver and \
                        self._is_good_component_blueprint(
                            label_var_name=label_var_name,
                            blueprint_obj=self):
                    label_var_names.append(label_var_name)

                    score_col_names[label_var_name] = \
                        component_blueprint_params.model.score.raw_score_col_prefix + label_var_name

                    lower_outlier_thresholds[label_var_name] = \
                        component_blueprint_params.data.label.lower_outlier_threshold

                    upper_outlier_thresholds[label_var_name] = \
                        component_blueprint_params.data.label.upper_outlier_threshold

        benchmark_metric_col_names = {}

        benchmark_metric_col_names_list = \
            ['pop__{}'.format(label_var_name)
             for label_var_name in label_var_names]

        for _global_or_indiv_prefix in (self._GLOBAL_PREFIX, self._INDIV_PREFIX, self._GLOBAL_OR_INDIV_PREFIX):
            benchmark_metric_col_names[_global_or_indiv_prefix] = {}

            for _raw_metric in (('n',) + self._RAW_METRICS):
                if (_global_or_indiv_prefix, _raw_metric) != (self._GLOBAL_OR_INDIV_PREFIX, 'n'):
                    benchmark_metric_col_names[_global_or_indiv_prefix][_raw_metric] = {}

                    for label_var_name in label_var_names:
                        benchmark_metric_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name] = \
                            benchmark_metric_col_name = \
                            _global_or_indiv_prefix + _raw_metric + '__' + label_var_name

                        benchmark_metric_col_names_list.append(benchmark_metric_col_name)

        err_mult_col_names = {}
        abs_err_mult_col_names = {}

        for _global_or_indiv_prefix in self._GLOBAL_OR_INDIV_PREFIXES:
            err_mult_col_names[_global_or_indiv_prefix] = {}
            abs_err_mult_col_names[_global_or_indiv_prefix] = {}

            for _raw_metric in self._RAW_METRICS:
                err_mult_col_names[_global_or_indiv_prefix][_raw_metric] = {}
                abs_err_mult_col_names[_global_or_indiv_prefix][_raw_metric] = {}

                for label_var_name in label_var_names:
                    err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name] = {}

                    for _sgn_prefix in self._SGN_PREFIXES:
                        err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][_sgn_prefix] = \
                            err_mult_col = \
                            _sgn_prefix + _global_or_indiv_prefix + self._ERR_MULT_PREFIXES[_raw_metric] + label_var_name

                        if _sgn_prefix == self._ABS_PREFIX:
                            abs_err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name] = err_mult_col

        id_col = self.params.data.id_col

        if isinstance(df, SparkADF):
            _is_adf = True

            benchmark_metrics_df = \
                df("SELECT \
                        DISTINCT({}) \
                    FROM \
                        this".format(id_col),
                   inheritCache=False,
                   inheritNRows=False) \
                .toPandas()

            for label_var_name in label_var_names:
                benchmark_metrics_df['pop__' + label_var_name] = \
                    len(self.params.benchmark_metrics[label_var_name][self._BY_ID_EVAL_KEY])

                for _raw_metric in (('n',) + self._RAW_METRICS):
                    _global_benchmark_metric_col_name = \
                        benchmark_metric_col_names[self._GLOBAL_PREFIX][_raw_metric][label_var_name]
                    benchmark_metrics_df.loc[:, _global_benchmark_metric_col_name] = \
                        self.params.benchmark_metrics[label_var_name][self._GLOBAL_EVAL_KEY][_raw_metric]

                    _indiv_benchmark_metric_col_name = \
                        benchmark_metric_col_names[self._INDIV_PREFIX][_raw_metric][label_var_name]
                    benchmark_metrics_df.loc[:, _indiv_benchmark_metric_col_name] = \
                        benchmark_metrics_df[id_col].map(
                            lambda _id:
                                self.params.benchmark_metrics[label_var_name][self._BY_ID_EVAL_KEY]
                                    .get(_id, {})
                                    .get(_raw_metric))

                    if _raw_metric != 'n':
                        _global_or_indiv_benchmark_metric_col_name = \
                            benchmark_metric_col_names[self._GLOBAL_OR_INDIV_PREFIX][_raw_metric][label_var_name]
                        benchmark_metrics_df.loc[:, _global_or_indiv_benchmark_metric_col_name] = \
                            benchmark_metrics_df[[_global_benchmark_metric_col_name, _indiv_benchmark_metric_col_name]] \
                            .max(axis='columns',
                                 skipna=True,
                                 level=None,
                                 numeric_only=True)

            SparkADF.create(
                data=benchmark_metrics_df.where(
                        cond=pandas.notnull(benchmark_metrics_df),
                        other=None,
                        inplace=False,
                        axis=None,
                        level=None,
                        errors='raise',
                        try_cast=False),
                schema=StructType(
                    [StructField(
                        name=id_col,
                        dataType=StringType(),
                        nullable=False,
                        metadata=None)] +
                    [StructField(
                        name=benchmark_metric_col_name,
                        dataType=DoubleType(),
                        nullable=True,
                        metadata=None)
                        for benchmark_metric_col_name in benchmark_metrics_df.columns[1:]]),
                alias=self._BENCHMARK_METRICS_ADF_ALIAS)

            df = df('SELECT \
                        this.*, \
                        {2} \
                    FROM \
                        this LEFT JOIN {0} \
                            ON this.{1} = {0}.{1}'
                .format(
                    self._BENCHMARK_METRICS_ADF_ALIAS,
                    id_col,
                    ', '.join('{}.{}'.format(self._BENCHMARK_METRICS_ADF_ALIAS, col)
                              for col in benchmark_metric_col_names_list)))

            col_exprs = []

            for label_var_name in label_var_names:
                score_col_name = score_col_names[label_var_name]

                _sgn_err_col_expr = \
                    pyspark.sql.functions.when(
                        condition=(df[label_var_name] > lower_outlier_thresholds[label_var_name])
                              and (df[label_var_name] < upper_outlier_thresholds[label_var_name]),
                        value=df[label_var_name] - df[score_col_name])

                for _global_or_indiv_prefix in self._GLOBAL_OR_INDIV_PREFIXES:
                    for _raw_metric in self._RAW_METRICS:
                        _sgn_err_mult_col_expr = \
                            _sgn_err_col_expr / \
                            df[benchmark_metric_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name]]

                        col_exprs += \
                            [_sgn_err_mult_col_expr
                                 .alias(err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._SGN_PREFIX]),

                             pyspark.sql.functions.abs(_sgn_err_mult_col_expr)
                                 .alias(err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._ABS_PREFIX]),

                             pyspark.sql.functions.when(df[label_var_name] < df[score_col_name], _sgn_err_mult_col_expr)
                                 .alias(err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._NEG_PREFIX]),

                             pyspark.sql.functions.when(df[label_var_name] > df[score_col_name], _sgn_err_mult_col_expr)
                                 .alias(err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._POS_PREFIX])]

            df = df.select('*', *col_exprs)

        else:
            _is_adf = False

            for label_var_name in label_var_names:
                score_col_name = score_col_names[label_var_name]

                _sgn_err_series = df[label_var_name] - df[score_col_name]

                _sgn_err_series.loc[
                    (df[label_var_name] <= lower_outlier_thresholds[label_var_name]) |
                    (df[label_var_name] >= upper_outlier_thresholds[label_var_name])] = numpy.nan

                _neg_chk_series = _sgn_err_series < 0
                _pos_chk_series = _sgn_err_series > 0

                for _raw_metric in (('n',) + self._RAW_METRICS):
                    _global_benchmark_metric_col_name = \
                        benchmark_metric_col_names[self._GLOBAL_PREFIX][_raw_metric][label_var_name]
                    df.loc[:, _global_benchmark_metric_col_name] = \
                        self.params.benchmark_metrics[label_var_name][self._GLOBAL_EVAL_KEY][_raw_metric]

                    _indiv_benchmark_metric_col_name = \
                        benchmark_metric_col_names[self._INDIV_PREFIX][_raw_metric][label_var_name]
                    df[_indiv_benchmark_metric_col_name] = \
                        df[id_col].map(
                            lambda id:
                                self.params.benchmark_metrics[label_var_name][self._BY_ID_EVAL_KEY]
                                    .get(id, {})
                                    .get(_raw_metric, numpy.nan))

                    if _raw_metric != 'n':
                        _global_or_indiv_benchmark_metric_col_name = \
                            benchmark_metric_col_names[self._GLOBAL_OR_INDIV_PREFIX][_raw_metric][label_var_name]
                        df.loc[:, _global_or_indiv_benchmark_metric_col_name] = \
                            df[[_global_benchmark_metric_col_name, _indiv_benchmark_metric_col_name]] \
                            .max(axis='columns',
                                 skipna=True,
                                 level=None,
                                 numeric_only=True)

                        for _global_or_indiv_prefix in self._GLOBAL_OR_INDIV_PREFIXES:
                            df[err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._SGN_PREFIX]] = \
                                _sgn_err_mult_series = \
                                _sgn_err_series / \
                                df[benchmark_metric_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name]]

                            df[err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._ABS_PREFIX]] = \
                                _sgn_err_mult_series.abs()

                            df.loc[_neg_chk_series,
                                   err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._NEG_PREFIX]] = \
                                _sgn_err_mult_series.loc[_neg_chk_series]

                            df.loc[_pos_chk_series,
                                   err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name][self._POS_PREFIX]] = \
                                _sgn_err_mult_series.loc[_pos_chk_series]

        n_label_vars = len(label_var_names)

        if _is_adf:
            _row_summ_col_exprs = []

            for _raw_metric in self._RAW_METRICS:
                for _global_or_indiv_prefix in self._GLOBAL_OR_INDIV_PREFIXES:
                    _abs_err_mult_col_names = \
                        list(abs_err_mult_col_names[_global_or_indiv_prefix][_raw_metric].values())

                    _row_summ_col_name_body = \
                        self._ABS_PREFIX + _global_or_indiv_prefix + self._ERR_MULT_COLS[_raw_metric]

                    _rowEuclNorm_summ_col_name = self._rowEuclNorm_PREFIX + _row_summ_col_name_body
                    _rowSumOfLog_summ_col_name = self._rowSumOfLog_PREFIX + _row_summ_col_name_body
                    _rowHigh_summ_col_name = self._rowHigh_PREFIX + _row_summ_col_name_body
                    _rowLow_summ_col_name = self._rowLow_PREFIX + _row_summ_col_name_body
                    _rowMean_summ_col_name = self._rowMean_PREFIX + _row_summ_col_name_body
                    _rowGMean_summ_col_name = self._rowGMean_PREFIX + _row_summ_col_name_body

                    if n_label_vars > 1:
                        _row_summ_col_exprs += \
                            ['POW({}, 0.5) AS {}'.format(
                                ' + '.join(
                                    'POW({} - 1, 2)'.format(_abs_err_mult_col_name)
                                    for _abs_err_mult_col_name in _abs_err_mult_col_names),
                                _rowEuclNorm_summ_col_name),

                                'LN({}) AS {}'.format(
                                    ' * '.join(_abs_err_mult_col_names),
                                    _rowSumOfLog_summ_col_name),

                                'GREATEST({}) AS {}'.format(
                                    ', '.join(_abs_err_mult_col_names),
                                    _rowHigh_summ_col_name),

                                'LEAST({}) AS {}'.format(
                                    ', '.join(_abs_err_mult_col_names),
                                    _rowLow_summ_col_name),

                                '(({}) / {}) AS {}'.format(
                                    ' + '.join(_abs_err_mult_col_names),
                                    n_label_vars,
                                    _rowMean_summ_col_name),

                                'POW({}, 1 / {}) AS {}'.format(
                                    ' * '.join(_abs_err_mult_col_names),
                                    n_label_vars,
                                    _rowGMean_summ_col_name)]

                    else:
                        _abs_err_mult_col_name = _abs_err_mult_col_names[0]

                        _row_summ_col_exprs += \
                            [df[_abs_err_mult_col_name].alias(_rowEuclNorm_summ_col_name),
                             pyspark.sql.functions.log(df[_abs_err_mult_col_name]).alias(_rowSumOfLog_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowHigh_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowLow_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowMean_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowGMean_summ_col_name)]

            return df.select('*', *_row_summ_col_exprs)

        else:
            if isinstance(df, ArrowADF):
                df = df.toPandas()

            for _raw_metric in self._RAW_METRICS:
                for _global_or_indiv_prefix in self._GLOBAL_OR_INDIV_PREFIXES:
                    _row_summ_col_name_body = \
                        self._ABS_PREFIX + _global_or_indiv_prefix + self._ERR_MULT_COLS[_raw_metric]

                    _rowEuclNorm_summ_col_name = self._rowEuclNorm_PREFIX + _row_summ_col_name_body
                    _rowSumOfLog_summ_col_name = self._rowSumOfLog_PREFIX + _row_summ_col_name_body
                    _rowHigh_summ_col_name = self._rowHigh_PREFIX + _row_summ_col_name_body
                    _rowLow_summ_col_name = self._rowLow_PREFIX + _row_summ_col_name_body
                    _rowMean_summ_col_name = self._rowMean_PREFIX + _row_summ_col_name_body
                    _rowGMean_summ_col_name = self._rowGMean_PREFIX + _row_summ_col_name_body

                    if n_label_vars > 1:
                        abs_err_mults_df = \
                            df[list(abs_err_mult_col_names[_global_or_indiv_prefix][_raw_metric].values())]

                        df[_rowEuclNorm_summ_col_name] = \
                            ((abs_err_mults_df - 1) ** 2).sum(
                                axis='columns',
                                skipna=True,
                                level=None,
                                numeric_only=None,
                                min_count=0) ** .5

                        _prod_series = \
                            abs_err_mults_df.product(
                                axis='columns',
                                skipna=False,
                                level=None,
                                numeric_only=True)

                        df[_rowSumOfLog_summ_col_name] = \
                            numpy.log(_prod_series)

                        df[_rowHigh_summ_col_name] = \
                            abs_err_mults_df.max(
                                axis='columns',
                                skipna=True,
                                level=None,
                                numeric_only=True)

                        df[_rowLow_summ_col_name] = \
                            abs_err_mults_df.min(
                                axis='columns',
                                skipna=True,
                                level=None,
                                numeric_only=True)

                        df[_rowMean_summ_col_name] = \
                            abs_err_mults_df.mean(
                                axis='columns',
                                skipna=True,
                                level=None,
                                numeric_only=True)

                        df[_rowGMean_summ_col_name] = \
                            _prod_series ** (1 / n_label_vars)

                    else:
                        _abs_err_mult_col_name = \
                            abs_err_mult_col_names[_global_or_indiv_prefix][_raw_metric][label_var_name]

                        df[_rowSumOfLog_summ_col_name] = \
                            numpy.log(df[_abs_err_mult_col_name])

                        df[_rowEuclNorm_summ_col_name] = \
                            df[_rowHigh_summ_col_name] = \
                            df[_rowLow_summ_col_name] = \
                            df[_rowMean_summ_col_name] = \
                            df[_rowGMean_summ_col_name] = \
                            df[_abs_err_mult_col_name]

            return df

    @classmethod
    def daily_err_mults(cls, df_w_err_mults, *label_var_names, **kwargs):
        id_col = kwargs.pop('id_col', 'id')
        time_col = kwargs.pop('time_col', 'date_time')

        clip = kwargs.pop('clip', 9)

        cols_to_agg = copy.copy(cls._ROW_ERR_MULT_SUMM_COLS)

        for label_var_name in label_var_names:
            if label_var_name in df_w_err_mults.columns:
                cols_to_agg += \
                    [(_sgn + _global_or_indiv_prefix + cls._ERR_MULT_PREFIXES[_metric] + label_var_name)
                     for _metric, _global_or_indiv_prefix, _sgn in
                     itertools.product(cls._RAW_METRICS, cls._GLOBAL_OR_INDIV_PREFIXES, cls._SGN_PREFIXES)]

        n_label_vars = len(label_var_names)

        if isinstance(df_w_err_mults, SparkADF):
            col_strs = []

            for col_name in cols_to_agg:
                assert col_name in df_w_err_mults.columns

                col_strs += \
                    ['PERCENTILE_APPROX(IF({0} IS NULL, NULL, GREATEST(LEAST({0}, {1}), -{1})), 0.5) AS {2}{0}'
                         .format(col_name, clip, cls._dailyMed_PREFIX),
                     'AVG(IF({0} IS NULL, NULL, GREATEST(LEAST({0}, {1}), -{1}))) AS {2}{0}'
                         .format(col_name, clip, cls._dailyMean_PREFIX),
                     'MAX(IF({0} IS NULL, NULL, GREATEST(LEAST({0}, {1}), -{1}))) AS {2}{0}'
                         .format(col_name, clip, cls._dailyMax_PREFIX),
                     'MIN(IF({0} IS NULL, NULL, GREATEST(LEAST({0}, {1}), -{1}))) AS {2}{0}'
                         .format(col_name, clip, cls._dailyMin_PREFIX)]

            for _global_or_indiv_prefix in cls._GLOBAL_OR_INDIV_PREFIXES:
                for _raw_metric in cls._RAW_METRICS:
                    for label_var_name in label_var_names:
                        if label_var_name in df_w_err_mults.columns:
                            _metric_col_name = _global_or_indiv_prefix + _raw_metric + '__' + label_var_name
                            cols_to_agg.append(_metric_col_name)
                            col_strs.append('AVG({0}) AS {0}'.format(_metric_col_name))

            adf = df_w_err_mults(
                'SELECT \
                    {0}, \
                    {1}, \
                    {2}, \
                    {3} \
                FROM \
                    this \
                GROUP BY \
                    {0}, \
                    {4}'
                .format(
                    id_col,
                    DATE_COL
                        if DATE_COL in df_w_err_mults.columns
                        else 'TO_DATE({}) AS {}'.format(time_col, DATE_COL),
                    ', '.join(col_strs),
                    ', '.join('FIRST_VALUE({0}) AS {0}'.format(col)
                              for col in set(df_w_err_mults.columns)
                                        .difference(
                                            [id_col, time_col, DATE_COL, MONTH_COL] +
                                            list(label_var_names) +
                                            [(_SupervisedBlueprintABC._DEFAULT_PARAMS.model.score.raw_score_col_prefix + label_var_name)
                                             for label_var_name in label_var_names] +
                                            cols_to_agg)),
                    DATE_COL
                        if DATE_COL in df_w_err_mults.columns
                        else 'TO_DATE({})'.format(time_col)),
                tCol=None)

            _row_summ_daily_summ_col_exprs = []

            for _raw_metric in cls._RAW_METRICS:
                for _global_or_indiv_prefix in cls._GLOBAL_OR_INDIV_PREFIXES:
                    for _daily_summ_prefix in cls._DAILY_SUMM_PREFIXES:
                        _row_summ_daily_summ_col_name_body = \
                            _daily_summ_prefix + cls._ABS_PREFIX + _global_or_indiv_prefix + cls._ERR_MULT_COLS[_raw_metric]

                        _daily_summ_abs_err_mult_col_names = \
                            [(_row_summ_daily_summ_col_name_body + '__' + label_var_name)
                             for label_var_name in label_var_names]

                        _rowEuclNorm_summ_daily_summ_col_name = \
                            cls._rowEuclNorm_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowSumOfLog_summ_daily_summ_col_name = \
                            cls._rowSumOfLog_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowHigh_summ_daily_summ_col_name = \
                            cls._rowHigh_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowLow_summ_daily_summ_col_name = \
                            cls._rowLow_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowMean_summ_daily_summ_col_name = \
                            cls._rowMean_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowGMean_summ_daily_summ_col_name = \
                            cls._rowGMean_PREFIX + _row_summ_daily_summ_col_name_body

                        if n_label_vars > 1:
                            _row_summ_daily_summ_col_exprs += \
                                ['POW({}, 0.5) AS {}'.format(
                                    ' + '.join(
                                        'POW({} - 1, 2)'.format(_daily_summ_abs_err_mult_col_name)
                                        for _daily_summ_abs_err_mult_col_name in _daily_summ_abs_err_mult_col_names),
                                    _rowEuclNorm_summ_daily_summ_col_name),

                                 'LN({}) AS {}'.format(
                                    ' * '.join(_daily_summ_abs_err_mult_col_names),
                                    _rowSumOfLog_summ_daily_summ_col_name),

                                 'GREATEST({}) AS {}'.format(
                                    ', '.join(_daily_summ_abs_err_mult_col_names),
                                    _rowHigh_summ_daily_summ_col_name),

                                 'LEAST({}) AS {}'.format(
                                    ', '.join(_daily_summ_abs_err_mult_col_names),
                                    _rowLow_summ_daily_summ_col_name),

                                 '(({}) / {}) AS {}'.format(
                                    ' + '.join(_daily_summ_abs_err_mult_col_names),
                                    n_label_vars,
                                    _rowMean_summ_daily_summ_col_name),

                                 'POW({}, 1 / {}) AS {}'.format(
                                    ' * '.join(_daily_summ_abs_err_mult_col_names),
                                    n_label_vars,
                                    _rowGMean_summ_daily_summ_col_name)]

                        else:
                            _daily_summ_abs_err_mult_col_name = _daily_summ_abs_err_mult_col_names[0]

                            _row_summ_daily_summ_col_exprs += \
                                [adf[_daily_summ_abs_err_mult_col_name].alias(_rowEuclNorm_summ_daily_summ_col_name),
                                 pyspark.sql.functions.log(adf[_daily_summ_abs_err_mult_col_name]).alias(_rowSumOfLog_summ_daily_summ_col_name),
                                 adf[_daily_summ_abs_err_mult_col_name].alias(_rowHigh_summ_daily_summ_col_name),
                                 adf[_daily_summ_abs_err_mult_col_name].alias(_rowLow_summ_daily_summ_col_name),
                                 adf[_daily_summ_abs_err_mult_col_name].alias(_rowMean_summ_daily_summ_col_name),
                                 adf[_daily_summ_abs_err_mult_col_name].alias(_rowGMean_summ_daily_summ_col_name)]

            return adf.select('*', *_row_summ_daily_summ_col_exprs)

        else:
            if df_w_err_mults[time_col].dtype != 'datetime64[ns]':
                df_w_err_mults[time_col] = pandas.DatetimeIndex(df_w_err_mults[time_col])

            def f(group_df):
                cols = [id_col, DATE_COL]

                _first_row = group_df.iloc[0]

                d = {id_col: _first_row[id_col],
                     DATE_COL: _first_row[time_col]}

                for _global_or_indiv_prefix in cls._GLOBAL_OR_INDIV_PREFIXES:
                    for _raw_metric in cls._RAW_METRICS:
                        for label_var_name in label_var_names:
                            if label_var_name in df_w_err_mults.columns:
                                _metric_col_name = _global_or_indiv_prefix + _raw_metric + '__' + label_var_name

                                d[_metric_col_name] = _first_row[_metric_col_name]

                                cols.append(_metric_col_name)

                for col_to_agg in cols_to_agg:
                    clipped_series = \
                        group_df[col_to_agg].clip(
                            lower=-clip,
                            upper=clip)

                    _dailyMed_agg_col = cls._dailyMed_PREFIX + col_to_agg
                    d[_dailyMed_agg_col] = clipped_series.median(skipna=True)

                    _dailyMean_agg_col = cls._dailyMean_PREFIX + col_to_agg
                    d[_dailyMean_agg_col] = clipped_series.mean(skipna=True)

                    _dailyMax_agg_col = cls._dailyMax_PREFIX + col_to_agg
                    d[_dailyMax_agg_col] = clipped_series.max(skipna=True)

                    _dailyMin_agg_col = cls._dailyMin_PREFIX + col_to_agg
                    d[_dailyMin_agg_col] = clipped_series.min(skipna=True)

                    cols += [_dailyMed_agg_col, _dailyMean_agg_col, _dailyMax_agg_col, _dailyMin_agg_col]

                return pandas.Series(d, index=cols)

            df = df_w_err_mults.groupby(
                    by=[df_w_err_mults[id_col], df_w_err_mults[time_col].dt.date],
                    axis='index',
                    level=None,
                    as_index=False,
                    sort=True,
                    group_keys=True,
                    squeeze=False).apply(f)

            for _raw_metric in cls._RAW_METRICS:
                for _global_or_indiv_prefix in cls._GLOBAL_OR_INDIV_PREFIXES:
                    for _daily_summ_prefix in cls._DAILY_SUMM_PREFIXES:
                        _row_summ_daily_summ_col_name_body = \
                            _daily_summ_prefix + cls._ABS_PREFIX + _global_or_indiv_prefix + cls._ERR_MULT_COLS[_raw_metric]

                        _rowEuclNorm_summ_daily_summ_col_name = \
                            cls._rowEuclNorm_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowSumOfLog_summ_daily_summ_col_name = \
                            cls._rowSumOfLog_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowHigh_summ_daily_summ_col_name = \
                            cls._rowHigh_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowLow_summ_daily_summ_col_name = \
                            cls._rowLow_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowMean_summ_daily_summ_col_name = \
                            cls._rowMean_PREFIX + _row_summ_daily_summ_col_name_body
                        _rowGMean_summ_daily_summ_col_name = \
                            cls._rowGMean_PREFIX + _row_summ_daily_summ_col_name_body

                        if n_label_vars > 1:
                            _daily_summ_abs_err_mults_df = \
                                df[[(_row_summ_daily_summ_col_name_body + '__' + label_var_name)
                                    for label_var_name in label_var_names]]

                            df[_rowEuclNorm_summ_daily_summ_col_name] = \
                                ((_daily_summ_abs_err_mults_df - 1) ** 2).sum(
                                    axis='columns',
                                    skipna=True,
                                    level=None,
                                    numeric_only=None,
                                    min_count=0) ** .5

                            _prod_series = \
                                _daily_summ_abs_err_mults_df.product(
                                    axis='columns',
                                    skipna=False,
                                    level=None,
                                    numeric_only=True)

                            df[_rowSumOfLog_summ_daily_summ_col_name] = \
                                numpy.log(_prod_series)

                            df[_rowHigh_summ_daily_summ_col_name] = \
                                _daily_summ_abs_err_mults_df.max(
                                    axis='columns',
                                    skipna=True,
                                    level=None,
                                    numeric_only=True)

                            df[_rowLow_summ_daily_summ_col_name] = \
                                _daily_summ_abs_err_mults_df.min(
                                    axis='columns',
                                    skipna=True,
                                    level=None,
                                    numeric_only=True)

                            df[_rowMean_summ_daily_summ_col_name] = \
                                _daily_summ_abs_err_mults_df.mean(
                                    axis='columns',
                                    skipna=True,
                                    level=None,
                                    numeric_only=True)

                            df[_rowGMean_summ_daily_summ_col_name] = \
                                _prod_series ** (1 / n_label_vars)

                        else:
                            _daily_summ_abs_err_mult_col_name = \
                                _row_summ_daily_summ_col_name_body + '__' + label_var_name

                            df[_rowSumOfLog_summ_daily_summ_col_name] = \
                                numpy.log(df[_daily_summ_abs_err_mult_col_name])

                            df[_rowEuclNorm_summ_daily_summ_col_name] = \
                                df[_rowHigh_summ_daily_summ_col_name] = \
                                df[_rowLow_summ_daily_summ_col_name] = \
                                df[_rowMean_summ_daily_summ_col_name] = \
                                df[_rowGMean_summ_daily_summ_col_name] = \
                                df[_daily_summ_abs_err_mult_col_name]

            return df

    @classmethod
    def ewma_daily_err_mults(cls, daily_err_mults_df, *daily_err_mult_summ_col_names, **kwargs):
        id_col = kwargs.pop('id_col', 'id')

        alpha = kwargs.pop('alpha', .168)

        daily_err_mult_summ_col_names = \
            list(daily_err_mult_summ_col_names) \
            if daily_err_mult_summ_col_names \
            else copy.copy(cls._DAILY_ERR_MULT_SUMM_COLS)

        daily_err_mults_df = \
            daily_err_mults_df[
                [id_col, DATE_COL] +
                daily_err_mult_summ_col_names]

        if not isinstance(daily_err_mults_df, pandas.DataFrame):
            daily_err_mults_df = daily_err_mults_df.toPandas()

        daily_err_mults_df.sort_values(
            by=[id_col, DATE_COL],
            axis='index',
            ascending=True,
            inplace=True,
            na_position='last')

        for _alpha in to_iterable(alpha):
            _ewma_prefix = cls._EWMA_PREFIX + '{:.3f}'.format(_alpha)[-3:] + '__'

            # ref: https://stackoverflow.com/questions/44417010/pandas-groupby-weighted-cumulative-sum
            daily_err_mults_df[
                [(_ewma_prefix + col_name)
                 for col_name in daily_err_mult_summ_col_names]] = \
                daily_err_mults_df.groupby(
                    by=id_col,
                        # Used to determine the groups for the groupby
                    axis='index',
                    level=None,
                        # If the axis is a MultiIndex (hierarchical), group by a particular level or levels
                    as_index=False,
                        # For aggregated output, return object with group labels as the index.
                        # Only relevant for DataFrame input. as_index=False is effectively SQL-style grouped output
                    sort=False,
                        # Sort group keys. Get better performance by turning this off.
                        # Note this does not influence the order of observations within each group.
                        # groupby preserves the order of rows within each group.
                    group_keys=False,
                        # When calling apply, add group keys to index to identify pieces
                    squeeze=False
                        # reduce the dimensionality of the return type if possible, otherwise return a consistent type
                )[daily_err_mult_summ_col_names] \
                .apply(
                    lambda df:
                        df.ewm(
                            com=None,
                            span=None,
                            halflife=None,
                            alpha=_alpha,
                            min_periods=0,
                            adjust=False,   # ref: http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows
                            ignore_na=True,
                            axis='index')
                        .mean())

        return daily_err_mults_df


# utility to create Blueprint from its params
def _blueprint_from_params(
        blueprint_params,
        aws_access_key_id=None,
        aws_secret_access_key=None,
        verbose=False):
    return import_obj(blueprint_params.__BlueprintClass__)(
            params=blueprint_params,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            verbose=verbose)


# utility to validate Blueprint params
def validate_blueprint_params(blueprint_or_blueprint_params):
    if isinstance(blueprint_or_blueprint_params, _BlueprintABC):
        blueprint = blueprint_or_blueprint_params
        blueprint_params = blueprint.params

    else:
        blueprint_params = blueprint_or_blueprint_params
        blueprint = \
            _blueprint_from_params(
                blueprint_params=blueprint_params,
                verbose=False)

    undeclared_params = \
        {k for k in blueprint_params.keys(all_nested=True)
         if not (k.startswith('data._cat_prep_cols_metadata.') or
                 k.startswith('data._num_prep_cols_metadata.'))} \
            .difference(_BLUEPRINT_PARAMS_ORDERED_LIST)

    component_blueprints = \
        blueprint_params.model.get('component_blueprints')

    if component_blueprints:
        component_blueprint_params = \
            {k for k in undeclared_params
             if k.startswith('model.component_blueprints.')}

        undeclared_params.difference_update(component_blueprint_params)

        _chk = all(validate_blueprint_params(component_blueprint)
                   for component_blueprint in component_blueprints.values())

    if undeclared_params:
        blueprint.stdout_logger.warning(
            msg='*** UNDECLARED PARAMS: {} ***'
                .format(undeclared_params))

        return False

    elif component_blueprints:
        return _chk

    else:
        return True


_LOADED_BLUEPRINTS = {}


# utility to load Blueprint from local or S3 dir path
def load(dir_path=None, s3_bucket=None, s3_dir_prefix=None,
         aws_access_key_id=None, aws_secret_access_key=None, s3_client=None,
         verbose=True):
    global _LOADED_BLUEPRINTS

    if dir_path in _LOADED_BLUEPRINTS:
        return _LOADED_BLUEPRINTS[dir_path]

    if verbose:
        logger = logging.getLogger(_LOGGER_NAME)
        logger.setLevel(logging.INFO)
        logger.addHandler(STDOUT_HANDLER)

    if ((s3_bucket and s3_dir_prefix) or (dir_path.startswith('s3://'))) \
            and ((aws_access_key_id and aws_secret_access_key) or s3_client):
        _from_s3 = True

        if s3_client is None:
            s3_client = \
                s3.client(
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key)

        elif not (aws_access_key_id and aws_secret_access_key):
            pass
            # TODO: create temp key pair from client
            # aws_access_key_id, aws_secret_access_key = ...

        if dir_path:
            s3_bucket, s3_dir_prefix = \
                dir_path.split('://')[1].split('/', 1)

        s3_parent_dir_prefix = \
            os.path.dirname(s3_dir_prefix)

        s3_file_key = \
            os.path.join(
                s3_dir_prefix,
                _BlueprintABC._DEFAULT_PARAMS.persist._file)

        if verbose:
            msg = 'Loading Blueprint Instance from S3 Path "s3://{}/{}..."'.format(s3_bucket, s3_file_key)
            logger.info(msg)

        _tmp_file_path = \
            os.path.join(
                tempfile.mkdtemp(),
                _TMP_FILE_NAME)

        s3_client.download_file(
            Bucket=s3_bucket,
            Key=s3_file_key,
            Filename=_tmp_file_path)

        params = joblib.load(filename=_tmp_file_path)

        params.uuid = os.path.basename(s3_dir_prefix)

        blueprint = \
            _blueprint_from_params(
                blueprint_params=params,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                verbose=verbose)

        if verbose:
            logger.info('{} {}'.format(msg, blueprint))

        fs.mv(from_path=_tmp_file_path,
              to_path=blueprint.file,
              is_dir=False,
              hdfs=False)

        s3_data_transforms_dir_prefix = \
            os.path.join(
                s3_dir_prefix,
                blueprint.params.data._transform_pipeline_dir)

        s3_data_transforms_dir_path = \
            's3://{}/{}'.format(
                s3_bucket, s3_data_transforms_dir_prefix)

        if 'Contents' in \
                s3_client.list_objects_v2(
                    Bucket=s3_bucket,
                    Prefix=s3_data_transforms_dir_prefix):
            if verbose:
                msg = 'Downloading Data Transforms for {} from S3 Path "{}"...'.format(blueprint, s3_data_transforms_dir_path)
                logger.info(msg)

            s3.sync(
                from_dir_path=s3_data_transforms_dir_path,
                to_dir_path=blueprint.data_transforms_dir,
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key,
                delete=True, quiet=False)

            if verbose:
                logger.info(msg + ' done!')

        if isinstance(blueprint, _SupervisedBlueprintABC):
            s3_models_dir_prefix = \
                os.path.join(
                    s3_dir_prefix,
                    blueprint.params.persist._models_dir)

            s3_models_dir_path = \
                's3://{}/{}'.format(
                    s3_bucket, s3_models_dir_prefix)

            if 'Contents' in \
                    s3_client.list_objects_v2(
                        Bucket=s3_bucket,
                        Prefix=s3_models_dir_prefix):
                if verbose:
                    msg = 'Downloading All Models Trained by {} from S3 Path "{}"...'.format(blueprint, s3_models_dir_path)
                    logger.info(msg)

                s3.sync(
                    from_dir_path=s3_models_dir_path,
                    to_dir_path=blueprint.models_dir,
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key,
                    delete=True, quiet=False)

                if verbose:
                    logger.info(msg + ' done!')

        if (params.persist.s3.bucket != s3_bucket) or (params.persist.s3.dir_prefix != s3_parent_dir_prefix):
            if verbose:
                msg = 'Re-Saving {} with Corrected S3 Path "s3://{}/{}"...'.format(blueprint, s3_bucket, s3_parent_dir_prefix)
                logger.info(msg)

            blueprint.params.persist.s3.bucket = s3_bucket
            blueprint.params.persist.s3.dir_prefix = s3_parent_dir_prefix
            blueprint.auth.aws.access_key_id = aws_access_key_id
            blueprint.auth.aws.secret_access_key = aws_secret_access_key

            blueprint.save()

            if verbose:
                logger.info(msg + ' done!')

    else:
        _from_s3 = False

        local_file_path = \
            os.path.join(
                dir_path,
                _BlueprintABC._DEFAULT_PARAMS.persist._file)

        if verbose:
            msg = 'Loading Blueprint Instance from Local Path "{}"...'.format(local_file_path)
            logger.info(msg)

        params = joblib.load(filename=local_file_path)

        params.uuid = os.path.basename(dir_path)

        blueprint = \
            _blueprint_from_params(
                blueprint_params=params,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                verbose=verbose)

        if verbose:
            logger.info('{} {}'.format(msg, blueprint))

    if 'component_blueprints' in blueprint.params.model:
        for label_var_name, component_blueprint_params in blueprint.params.model.component_blueprints.items():
            # legacy fix
            _component_blueprint_uuid_prefix = \
                '{}---{}'.format(params.uuid, label_var_name)

            if not component_blueprint_params.uuid.startswith(_component_blueprint_uuid_prefix):
                component_blueprint_params.uuid = _component_blueprint_uuid_prefix

            component_blueprint = \
                _blueprint_from_params(
                    blueprint_params=component_blueprint_params,
                    verbose=verbose)

            assert component_blueprint.params.uuid == component_blueprint_params.uuid, \
                '*** {} ***'.format(component_blueprint.params.uuid)

            # force load to test existence / sync down component blueprints from S3 if necessary
            try:
                load(dir_path=component_blueprint.path,
                     aws_access_key_id=blueprint.auth.aws.access_key_id,
                     aws_secret_access_key=blueprint.auth.aws.secret_access_key,
                     verbose=verbose)

            except:
                if _from_s3:
                    component_blueprint.params.persist.s3.bucket = s3_bucket
                    component_blueprint.params.persist.s3.dir_prefix = s3_parent_dir_prefix
                    component_blueprint.auth.aws.access_key_id = aws_access_key_id
                    component_blueprint.auth.aws.secret_access_key = aws_secret_access_key

                    try:
                        load(dir_path=component_blueprint.path,
                             aws_access_key_id=blueprint.auth.aws.access_key_id,
                             aws_secret_access_key=blueprint.auth.aws.secret_access_key,
                             verbose=verbose)

                        component_blueprint.save()

                    except:
                        blueprint.stdout_logger.warning(
                            msg='*** COMPONENT BLUEPRINT {} FAILS TO LOAD ***'
                                .format(component_blueprint))

                else:
                    blueprint.stdout_logger.warning(
                        msg='*** COMPONENT BLUEPRINT {} FAILS TO LOAD ***'
                            .format(component_blueprint))

    if dir_path:
        _LOADED_BLUEPRINTS[dir_path] = blueprint

    return blueprint
