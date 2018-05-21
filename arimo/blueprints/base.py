# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals

import abc
import argparse
import copy
import joblib
import logging
import math
import os
import pandas
import tempfile
import time
import tqdm
import uuid

import arimo.backend
import arimo.eval.metrics
from arimo.util import clean_str, clean_uuid, date_time, fs, import_obj, Namespace
from arimo.util.aws import s3
from arimo.util.date_time import _PRED_VARS_INCL_T_AUX_COLS, _PRED_VARS_INCL_T_CAT_AUX_COLS, _PRED_VARS_INCL_T_NUM_AUX_COLS
from arimo.util.log import STDOUT_HANDLER
from arimo.util.pkl import COMPAT_PROTOCOL, COMPAT_COMPRESS, MAX_COMPRESS_LVL, PKL_EXT
from arimo.util.types.spark_sql import _NUM_TYPES, _STR_TYPE
import arimo.debug

from .mixins.anom import PPPAnalysesMixIn
from .mixins.eval import RegrEvalMixIn


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

        # set local dir storing all models for Blueprint
        if self.params.persist._models_dir != self._DEFAULT_PARAMS.persist._models_dir:
            self.params.persist._models_dir = self._DEFAULT_PARAMS.persist._models_dir

        self.models_dir = \
            os.path.join(
                self.dir,
                self.params.persist._models_dir)

        # set BlueprintedModelClass
        self.__BlueprintedModelClass__ = \
            import_obj(
                self.params.get('__BlueprintedModelClass__',
                                'arimo.blueprints.base.BlueprintedKerasModel'))

        self.params.__BlueprintedModelClass__ = \
            self.__BlueprintedModelClass__.__qual_name__()

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
    from arimo.blueprints.base import _BlueprintABC, _blueprint_from_params

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
            component_blueprint = \
                _blueprint_from_params(
                    blueprint_params=component_blueprint_params,
                    verbose=verbose)

            # legacy fix
            component_blueprint.params.uuid = component_blueprint_params.uuid

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
        return '{} Model v{}'.format(self.blueprint, self.ver)

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

            self._obj = \
                model_factory(
                    params=self.blueprint.params,
                    **model_factory_params)

            self.stdout_logger.info(msg + ' done!')

            # if Keras model, then print a summary for ease of review
            if isinstance(self._obj, arimo.backend.keras.models.Model):
                self.summary()

        return getattr(self._obj, item)

    def copy(self):
        # create a new version with Blueprint's Model Factory
        model = type(self)(
            blueprint=self.blueprint,
            ver=None)

        # use existing model object to overwrite recipe's initial model
        model._obj = self._obj

        # save new Model
        model.save()

        return model


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
                delete=True, quiet=False)

            if verbose:
                self.blueprint.stdout_logger.info(msg + ' done!')

        if verbose:
            self.stdout_logger.info(message + ' done!')


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

    def __repr__(self):
        return '{} Instance "{}" (Label: "{}")'.format(
            self.__qual_name__(),
            self.params.uuid,
            self.params.data.label.var)


@_docstr_blueprint
class _DLSupervisedBlueprintABC(_SupervisedBlueprintABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_PARAMS = \
        copy.deepcopy(
            _SupervisedBlueprintABC._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        model=Namespace(
            train=Namespace(
                n_samples_max_multiple_of_data_size=9,
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

    DEFAULT_MODEL_TRAIN_MAX_GEN_QUEUE_SIZE = 99   # sufficient to keep CPUs busy while feeding into GPU
    DEFAULT_MODEL_TRAIN_N_WORKERS = 9   # 2-3x No. of CPUs

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
class _PPPBlueprintABC(_BlueprintABC, PPPAnalysesMixIn):
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

    def train(self, *args, **kwargs):
        __gen_queue_size__ = \
            kwargs.pop(
                '__gen_queue_size__',
                _DLSupervisedBlueprintABC.DEFAULT_MODEL_TRAIN_MAX_GEN_QUEUE_SIZE)

        __n_workers__ = \
            kwargs.pop(
                '__n_workers__',
                _DLSupervisedBlueprintABC.DEFAULT_MODEL_TRAIN_N_WORKERS)

        assert __n_workers__, '*** __n_workers__ = {} ***'.format(__n_workers__)

        __multiproc__ = kwargs.pop('__multiproc__', True)

        # whether to retrain component Blueprinted models
        __retrain_components__ = kwargs.pop('__retrain_components__', False)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        _medianFill = kwargs.pop('_medianFill', False)

        component_labeled_adfs = \
            self.prep_data(
                __mode__=self._TRAIN_MODE,
                verbose=verbose,
                _medianFill=_medianFill,
                *args, **kwargs)

        models = Namespace()

        for label_var_name, component_labeled_adf in component_labeled_adfs.items():
            blueprint_params = self.params.model.component_blueprints[label_var_name]

            blueprint_params.uuid = \
                '{}---{}---{}'.format(
                    self.params.uuid,
                    label_var_name,
                    uuid.uuid4())

            if __retrain_components__:
                blueprint_params.model.ver = None

                blueprint_params.data.label.lower_outlier_threshold = \
                    blueprint_params.data.label.upper_outlier_threshold = \
                    None

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
                        verbose=verbose,
                        _medianFill=_medianFill)

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

            # *** COPY DataTransforms AFTER model training so that __first_train__ is correctly detected ***
            if not os.path.isdir(blueprint.data_transforms_dir):
                fs.cp(from_path=self.data_transforms_dir,
                      to_path=blueprint.data_transforms_dir,
                      hdfs=False,
                      is_dir=True)

        # save Blueprint, with updated component blueprint params
        self.save()

        return Namespace(
            blueprint=self,
            models=models)

    def eval(self, *args, **kwargs):
        # whether to exclude outlying labels from eval
        __excl_outliers__ = kwargs.pop('__excl_outliers__', False)

        # whether to cache data at certain stages
        __cache_vector_data__ = kwargs.pop('__cache_vector_data__', False)
        __cache_tensor_data__ = kwargs.pop('__cache_tensor_data__', False)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        adf = self.score(
            __mode__=self._EVAL_MODE,
            __cache_vector_data__=__cache_vector_data__,
            __cache_tensor_data__=__cache_tensor_data__,
            verbose=verbose,
            *args, **kwargs)

        label_var_names = []

        for label_var_name, blueprint_params in self.params.model.component_blueprints.items():
            if (label_var_name in adf.columns) and blueprint_params.model.ver:
                label_var_names.append(label_var_name)

        # cache to calculate multiple metrics quickly
        if len(label_var_names) > 1:
            adf.cache(
                eager=True,
                verbose=verbose)

        eval_metrics = {}

        id_col = self.params.data.id_col
        id_col_type_is_str = (adf.type(id_col) == _STR_TYPE)

        for label_var_name in label_var_names:
            score_col_name = blueprint_params.model.score.raw_score_col_prefix + label_var_name

            _per_label_adf = \
                adf[id_col, score_col_name, label_var_name] \
                    .filter(
                        condition='({} IS NOT NULL) AND ({} IS NOT NULL)'
                            .format(label_var_name, score_col_name),
                        alias=adf.alias + '__toEval__' + label_var_name)

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
                    ('BETWEEN {} AND {}'
                        .format(
                            blueprint_params.data.label.lower_outlier_threshold,
                            blueprint_params.data.label.upper_outlier_threshold)
                     if _upper_outlier_threshold_applicable
                     else '>= {}'.format(blueprint_params.data.label.lower_outlier_threshold)) \
                    if _lower_outlier_threshold_applicable \
                    else '<= {}'.format(blueprint_params.data.label.upper_outlier_threshold)

                if arimo.debug.ON:
                    self.stdout_logger.debug(
                        msg='*** EVAL: CONDITION ROBUST TO OUTLIER LABELS: {} {} ***\n'
                            .format(label_var_name, _outlier_robust_condition))

                _per_label_adf.filter(
                    condition='{} {}'.format(
                        label_var_name,
                        _outlier_robust_condition),
                    inplace=True)

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

            ids = _per_label_adf(
                    "SELECT \
                        DISTINCT({}) \
                    FROM \
                        this".format(id_col),
                    inheritCache=False,
                    inheritNRows=False) \
                .toPandas()[id_col]

            for id in tqdm.tqdm(ids):
                _per_label_per_id_adf = \
                    _per_label_adf \
                        .filter(
                            condition="{} = {}"
                                .format(
                                    id_col,
                                    "'{}'".format(id))
                                        if id_col_type_is_str
                                        else id) \
                        .drop(
                            id_col,
                            alias=_per_label_adf.alias + '__' + clean_str(id))

                # cache to calculate multiple metrics quickly
                _per_label_per_id_adf.cache(
                    eager=True,
                    verbose=False)

                eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][id] = \
                    dict(n=arimo.eval.metrics.n(_per_label_per_id_adf))

                for evaluator in evaluators:
                    eval_metrics[label_var_name][self._BY_ID_EVAL_KEY][id][evaluator.name] = \
                        evaluator(_per_label_per_id_adf)

                _per_label_per_id_adf.unpersist()

            _per_label_adf.unpersist()

        adf.unpersist()

        return eval_metrics

    def save_benchmark_metrics(self, *args, **kwargs):
        self.params.benchmark_metrics = \
            self.eval(
                __excl_outliers__=True,
                *args, **kwargs)

        self.save()
