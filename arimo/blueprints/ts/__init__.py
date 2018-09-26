from __future__ import absolute_import, division, print_function

import abc
import copy
import itertools
import logging
import numpy
import os
import pandas
import shutil
import tqdm
import uuid

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructField, StructType

import arimo.backend
from arimo.data.distributed import DDF
import arimo.eval.metrics
from arimo.util import clean_str, clean_uuid, fs, Namespace
from arimo.util.decor import _docstr_verbose
from arimo.util.dl import MASK_VAL
from arimo.util.log import STDOUT_HANDLER
from arimo.util.types.spark_sql import _NUM_TYPES, _STR_TYPE
import arimo.debug

from ..base import _docstr_blueprint, _DLSupervisedBlueprintABC, RegrEvalMixIn, \
    BlueprintedArimoDLModel, BlueprintedKerasModel


@_docstr_blueprint
class _TimeSerDLSupervisedBlueprintABC(_DLSupervisedBlueprintABC):
    __metaclass__ = abc.ABCMeta

    _DEFAULT_PARAMS = \
        copy.deepcopy(
            _DLSupervisedBlueprintABC._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        # input series and prediction horizon lengths
        min_input_ser_len=1,   # min number of time slices used as input series
        max_input_ser_len=1,   # max number of time slices used as input series
        pred_horizon_len=0,   # number of time slices to predict ahead

        __metadata__={
            'min_input_ser_len': Namespace(
                label='Min History Length',
                description="Min No. of Time Slices of Each Entity's History to be Considered for Use in Prediction",
                type='int',
                default=1),

            'max_input_ser_len': Namespace(
                label='Max History Length',
                description="Max No. of Time Slices of Each Entity's History to be Used in Prediction",
                type='int',
                default=1),

            'pred_horizon_len': Namespace(
                label='Prediction Horizon Length',
                description='Prediction Horizon Length',
                type='timedelta',
                default=0,
                tags=[])})

    _EVAL_SCORE_ADF_ALIAS = '__EvalScored__'

    @_docstr_verbose
    def score(self, *args, **kwargs):
        # whether scoring for eval purposes
        __eval__ = kwargs.pop(self._MODE_ARG, None) == self._EVAL_MODE
        if __eval__:
            __excl_outliers__ = kwargs.pop('__excl_outliers__', False)

        adf = kwargs.pop('__adf__', None)
        model = kwargs.pop('__model__', None)

        # whether to score certain time periods only
        __score_filter__ = kwargs.pop('__score_filter__', None)

        # scoring batch size
        __batch_size__ = kwargs.pop('__batch_size__', 32)

        # whether to cache data at certain stages
        __cache_vector_data__ = kwargs.pop('__cache_vector_data__', False)
        __cache_tensor_data__ = kwargs.pop('__cache_tensor_data__', False)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        if __eval__:
            assert adf and model

            if arimo.debug.ON:
                model.stdout_logger.debug(
                    msg='*** SCORING IN EVAL MODE ***')

            label_var = \
                self.params.data.label.var \
                if self.params.data.label._int_var is None \
                else self.params.data.label._int_var

            if __excl_outliers__:
                label_var_type = adf.type(label_var)
                assert (label_var_type.startswith('decimal') or (label_var_type in _NUM_TYPES))

                _outlier_robust_condition = \
                    '>= {}'.format(self.params.data.label.lower_outlier_threshold) \
                    if pandas.isnull(self.params.data.label.upper_outlier_threshold) \
                    else ('<= {}'.format(self.params.data.label.upper_outlier_threshold)
                          if pandas.isnull(self.params.data.label.lower_outlier_threshold)
                          else 'BETWEEN {} AND {}'.format(
                                    self.params.data.label.lower_outlier_threshold,
                                    self.params.data.label.upper_outlier_threshold))

                if arimo.debug.ON:
                    model.stdout_logger.debug(
                        msg='*** SCORE FOR EVAL: CONDITION ROBUST TO OUTLIER LABELS: {} {} ***'
                            .format(label_var, _outlier_robust_condition))

        else:
            adf = self.prep_data(
                    __mode__=self._SCORE_MODE,
                    verbose=verbose,
                    *args, **kwargs)

            model = self.model(
                ver=self.params.model.ver
                    if self.params.model.ver
                    else 'latest')

            label_var = \
                self.params.data.label.var \
                if self.params.data.label._int_var is None \
                else self.params.data.label._int_var

        assert isinstance(adf, DDF) and adf.alias

        if __cache_vector_data__:   # *** cache here to potentially speed up window funcs creating sequences below ***
            adf.cache(
                eager=True,
                verbose=verbose)

        score_adf = \
            adf._prepareArgsForSampleOrGenOrPred(
                (self.params.data._prep_vec_col,
                 - self.params.max_input_ser_len + 1, 0),
                n=None,
                fraction=None,
                withReplacement=False,
                seed=None,
                anon=False,
                collect=False,
                pad=MASK_VAL,
                filter=
                    '({} >= {}) {}'.format(
                        adf._T_ORD_IN_CHUNK_COL, self.params.min_input_ser_len,
                        'AND ({} {})'.format(
                            'LEAD({}, {}) OVER (PARTITION BY {}, {} ORDER BY {})'.format(
                                label_var, self.params.pred_horizon_len,
                                self.params.data.id_col, adf._T_CHUNK_COL, adf._T_ORD_COL)
                            if self.params.pred_horizon_len
                            else label_var,
                            _outlier_robust_condition
                                if __excl_outliers__
                                else 'IS NOT NULL')
                        if __eval__
                        else ('AND ({})'.format(__score_filter__)
                              if __score_filter__
                              else '')),
                keepOrigRows=False) \
            .adf   # TODO: keepOrigRows should ideally = not __eval__, but set to False here

        if arimo.debug.ON:
            model.stdout_logger.debug(
                msg='*** SCORE: PREPARED DDF: {} {} ***'
                    .format(adf, adf.columns))

        id_col = str(self.params.data.id_col)
        time_chunk_col = score_adf._T_CHUNK_COL
        time_ord_in_chunk_col = score_adf._T_ORD_IN_CHUNK_COL
        prep_vec_col = self.params.data._prep_vec_col
        prep_vec_size = self.params.data._prep_vec_size
        prep_vec_over_time_col = \
            next(col for col in score_adf.columns
                     if prep_vec_col in col)
        max_input_ser_len = self.params.max_input_ser_len

        if isinstance(model, BlueprintedArimoDLModel):
            model_path = _model_path = model.dir

            if fs._ON_LINUX_CLUSTER_WITH_HDFS and (model_path not in self._MODEL_PATHS_ON_SPARK_WORKER_NODES):
                fs.put(
                    from_local=model_path,
                    to_hdfs=model_path,
                    is_dir=True,
                    _mv=False)

                self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_path] = model_path

        else:
            assert isinstance(model, BlueprintedKerasModel)

            model_path = os.path.join(model.dir, self.params.model._persist.file)

            if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                if model_path not in self._MODEL_PATHS_ON_SPARK_WORKER_NODES:
                    _tmp_local_file_name = \
                        str(uuid.uuid4())

                    _tmp_local_file_path = \
                        os.path.join(
                            '/tmp',
                            _tmp_local_file_name)

                    shutil.copyfile(
                        src=model_path,
                        dst=_tmp_local_file_path)

                    arimo.backend.spark.sparkContext.addFile(
                        path=_tmp_local_file_path,
                        recursive=False)

                    self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_path] = \
                        _tmp_local_file_name   # SparkFiles.get(filename=_tmp_local_file_name)

                _model_path = self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_path]

            else:
                _model_path = model_path

        raw_score_col = self.params.model.score.raw_score_col_prefix + self.params.data.label.var
        
        def batch(row_iterator_in_partition):
            rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))
            
            while rows:
                yield \
                    [row[id_col] for row in rows], \
                    [row[time_chunk_col] for row in rows], \
                    [row[time_ord_in_chunk_col] for row in rows], \
                    numpy.vstack(
                        numpy.expand_dims(
                            numpy.vstack(
                                [numpy.zeros(
                                    (max_input_ser_len - len(row[prep_vec_over_time_col]),
                                     prep_vec_size))] +
                                [r[prep_vec_col].toArray()
                                 for r in row[prep_vec_over_time_col]]),
                            axis=0)
                        for row in rows)

                rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

        rdd = score_adf.rdd.mapPartitions(batch)

        if __cache_tensor_data__:
            rdd.cache()
            rdd.count()

        n_classes = self.params.data.label._n_classes
        binary = (n_classes == 2)

        if isinstance(model, BlueprintedArimoDLModel):
            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                from arimo.util.dl import _load_arimo_dl_model

                return [(i, chunk, t_ord_in_chunk,
                         float(score[0])
                            if (n_classes is None) or binary
                            else score.tolist())
                        for i, chunk, t_ord_in_chunk, score in
                            zip(tup[0], tup[1], tup[2],
                                _load_arimo_dl_model(
                                        dir_path=_model_path,
                                        hdfs=cluster)
                                    .predict(
                                        data=tup[3],
                                        input_tensor_transform_fn=None,
                                        batch_size=__batch_size__))]

        else:
            assert isinstance(model, BlueprintedKerasModel)

            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                if cluster:
                    try:
                        from arimo.util.dl import _load_keras_model

                    except ImportError:
                        from dl import _load_keras_model

                else:
                    from arimo.util.dl import _load_keras_model

                return [(i, chunk, t_ord_in_chunk,
                         float(score[0])
                            if (n_classes is None) or binary
                            else score.tolist())
                        for i, chunk, t_ord_in_chunk, score in
                            zip(tup[0], tup[1], tup[2],
                                _load_keras_model(
                                        file_path=_model_path)
                                    .predict(
                                        x=tup[3],
                                        batch_size=__batch_size__,
                                        verbose=0))]

        score_adf = \
            DDF.create(
                data=rdd.flatMap(score),
                schema=StructType(
                    [StructField(
                        name=id_col,
                        dataType=adf._schema[id_col].dataType,
                        nullable=True,
                        metadata=None),
                     StructField(
                        name=time_chunk_col,
                        dataType=IntegerType(),
                        nullable=True,
                        metadata=None),
                     StructField(
                        name=time_ord_in_chunk_col,
                        dataType=IntegerType(),
                        nullable=True,
                        metadata=None),
                     StructField(
                        name=raw_score_col,
                        dataType=
                            DoubleType()
                            if (n_classes is None) or binary
                            else ArrayType(
                                elementType=DoubleType(),
                                containsNull=False),
                        nullable=True,
                        metadata=None)]),
                samplingRatio=None,
                verifySchema=False)

        if __eval__:
            return score_adf

        else:
            score_adf.alias = self._SCORE_ADF_ALIAS

            return adf(
                'SELECT \
                    {4}, \
                    {0}.{5} AS {5} \
                FROM \
                    this LEFT JOIN {0} \
                        ON this.{1} = {0}.{1} AND \
                           this.{2} = {0}.{2} AND \
                           this.{3} = {0}.{3}'
                    .format(
                        self._SCORE_ADF_ALIAS,
                        id_col,
                        time_chunk_col,
                        time_ord_in_chunk_col,
                        ', '.join('this.{0} AS {0}'.format(col)
                                  for col in adf.columns
                                  if col != prep_vec_col),
                        raw_score_col))

    def eval(self, *args, **kwargs):
        _lower_outlier_threshold_applicable = \
            pandas.notnull(self.params.data.label.lower_outlier_threshold)

        _upper_outlier_threshold_applicable = \
            pandas.notnull(self.params.data.label.upper_outlier_threshold)

        __excl_outliers__ = \
            kwargs.pop('__excl_outliers__', False) and \
            isinstance(self, RegrEvalMixIn) and \
            self.params.data.label.excl_outliers and \
            self.params.data.label.outlier_tails and \
            self.params.data.label.outlier_tail_proportion and \
            (self.params.data.label.outlier_tail_proportion < .5) and \
            (_lower_outlier_threshold_applicable or _upper_outlier_threshold_applicable)

        # whether to cache data at certain stages
        __cache_vector_data__ = kwargs.pop('__cache_vector_data__', False)
        __cache_tensor_data__ = kwargs.pop('__cache_tensor_data__', False)

        __batch_size__ = kwargs.pop('__batch_size__', 500)

        __metrics__ = \
            kwargs.pop(
                '__metrics__',
                self.eval_metrics)

        n_classes = self.params.data.label._n_classes

        if n_classes:
            __class_thresholds__ = \
                kwargs.pop(
                    '__class_thresholds__',
                    n_classes * (1 / n_classes,))

            if isinstance(__class_thresholds__, float):
                __class_thresholds__ = (1 - __class_thresholds__, __class_thresholds__)

        verbose = kwargs.pop('verbose', True)

        if arimo.debug.ON:
            logger = logging.getLogger(self._logger_name)
            logger.setLevel(logging.DEBUG)
            logger.addHandler(STDOUT_HANDLER)

        adf = self.prep_data(
                __mode__=self._EVAL_MODE,
                verbose=verbose,
                *args, **kwargs)

        assert isinstance(adf, DDF) and adf.alias

        model = self.model(
            ver=self.params.model.ver
                if self.params.model.ver
                else 'latest')

        if arimo.debug.ON:
            model.stdout_logger.debug(
                msg='*** EVAL: POST-PREP_XY DDF: {} {} ***'
                    .format(adf, adf.columns))

        score_adf = \
            self.score(
                __mode__=self._EVAL_MODE,
                __excl_outliers__=__excl_outliers__,
                __adf__=adf,
                __cache_vector_data__=__cache_vector_data__,
                __cache_tensor_data__=__cache_tensor_data__,
                __model__=model,
                __score_filter__=None,   # TODO: examine whether this should be relaxed to eval certain time slices only
                __batch_size__=__batch_size__)

        score_adf.alias = self._EVAL_SCORE_ADF_ALIAS

        if arimo.debug.ON:
            model.stdout_logger.debug(
                msg='*** EVAL: SCORED DDF: {} {} ***'
                    .format(score_adf, score_adf.columns))

        label_var = \
            self.params.data.label._int_var \
            if self.params.data.label._int_var \
            else self.params.data.label.var

        if __excl_outliers__:
            label_var_type = adf.type(label_var)
            assert (label_var_type.startswith('decimal') or (label_var_type in _NUM_TYPES))

            _outlier_robust_condition = \
                ('BETWEEN {} AND {}'
                    .format(
                        self.params.data.label.lower_outlier_threshold,
                        self.params.data.label.upper_outlier_threshold)
                 if _upper_outlier_threshold_applicable
                 else '>= {}'.format(self.params.data.label.lower_outlier_threshold)) \
                if _lower_outlier_threshold_applicable \
                else '<= {}'.format(self.params.data.label.upper_outlier_threshold)

            if arimo.debug.ON:
                logger.debug(
                    msg='*** EVAL: CONDITION ROBUST TO OUTLIER LABELS: {} {} ***\n'
                        .format(label_var, _outlier_robust_condition))

        raw_score_col = self.params.model.score.raw_score_col_prefix + self.params.data.label.var

        adf('SELECT \
                {0}.{1} AS {1}, \
                this.{2} AS {2} \
            FROM \
                this JOIN {0} \
                    ON (this.{2} {3}) AND \
                       (this.{4} = {0}.{4}) AND \
                       (this.{5} = {0}.{5}) AND \
                       (this.{6} = {0}.{6} + {7})'
                .format(
                    self._EVAL_SCORE_ADF_ALIAS,
                    raw_score_col,
                    label_var,
                    _outlier_robust_condition
                        if __excl_outliers__
                        else 'IS NOT NULL',
                    self.params.data.id_col,
                    adf._T_CHUNK_COL,
                    adf._T_ORD_IN_CHUNK_COL,
                    self.params.pred_horizon_len),
            inplace=True)

        adf.alias += self._EVAL_SUFFIX

        adf.cache(
            eager=True,
            verbose=verbose)

        metrics = \
            {self._GLOBAL_EVAL_KEY: dict(n=arimo.eval.metrics.n(adf)),
             self._BY_ID_EVAL_KEY: {}}

        evaluators = []

        for metric_name in __metrics__:
            metric_class = getattr(arimo.eval.metrics, metric_name)

            evaluator = \
                (metric_class(
                    label_col=label_var,
                    n_classes=n_classes,
                    labels=self.params.data.label.get('_strings'))
                 if metric_name == 'Prevalence'
                 else metric_class(
                    label_col=label_var,
                    score_col=raw_score_col,
                    n_classes=self.params.data.label._n_classes,
                    labels=self.params.data.label.get('_strings'))) \
                if n_classes \
                else metric_class(
                    label_col=label_var,
                    score_col=raw_score_col)

            evaluators.append(evaluator)

            metrics[evaluator.name] = \
                evaluator(adf, *__class_thresholds__) \
                if n_classes and (metric_name != 'Prevalence') \
                else evaluator(adf)

        id_col = self.params.data.id_col

        if id_col in adf.columns:
            id_col_type_is_str = (adf.type(id_col) == _STR_TYPE)

            ids = adf("SELECT \
                        DISTINCT({}) \
                      FROM \
                        this".format(id_col),
                      inheritCache=False,
                      inheritNRows=False) \
                .toPandas()[id_col]

            for id in tqdm.tqdm(ids.loc[ids.notnull()]):
                _per_id_adf = \
                    adf.filter(
                        condition="{} = {}"
                            .format(
                                id_col,
                                "'{}'".format(id))
                                    if id_col_type_is_str
                                    else id) \
                    .drop(
                        id_col,
                        alias=(adf.alias + '__' + clean_str(clean_uuid(id)))
                            if arimo.debug.ON
                            else None)

                # cache to calculate multiple metrics quickly
                _per_id_adf.cache(
                    eager=True,
                    verbose=False)

                metrics[self._BY_ID_EVAL_KEY][id] = \
                    dict(n=arimo.eval.metrics.n(_per_id_adf))

                for evaluator in evaluators:
                    metrics[self._BY_ID_EVAL_KEY][id][evaluator.name] = \
                        evaluator(_per_id_adf)

                _per_id_adf.unpersist()

        adf.unpersist()

        return metrics
