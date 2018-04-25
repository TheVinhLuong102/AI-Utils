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

from pyspark import SparkFiles
from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType

import arimo.backend
from arimo.blueprints.base import _docstr_blueprint, _SupervisedBlueprintABC, _DLSupervisedBlueprintABC, KerasModel
from arimo.df.spark import ADF
import arimo.eval.metrics
from arimo.util import fs
from arimo.util.log import STDOUT_HANDLER
from arimo.util.spark_sql_types import _STR_TYPE

from arimo.util.spark_sql_types import _NUM_TYPES, _VECTOR_TYPE
import arimo.debug

from ..mixins.data_prep import LabeledDataPrepMixIn
from ..mixins.eval import RegrEvalMixIn


@_docstr_blueprint
class _CrossSectSupervisedBlueprintABC(LabeledDataPrepMixIn, _SupervisedBlueprintABC):
    __metaclass__ = abc.ABCMeta

    def eval(self, *args, **kwargs):
        # whether to exclude outlying labels
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

        # scoring batch size
        __batch_size__ = kwargs.pop('__batch_size__', 1000)

        # eval metrics to compute
        __metrics__ = \
            kwargs.pop(
                '__metrics__',
                self.eval_metrics)

        # incoporate __class_thresholds__ if passed
        n_classes = self.params.data.label._n_classes

        if n_classes:
            __class_thresholds__ = \
                kwargs.pop(
                    '__class_thresholds__',
                    n_classes * (1 / n_classes,))

            if isinstance(__class_thresholds__, float):
                __class_thresholds__ = (1 - __class_thresholds__, __class_thresholds__)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        # prep data
        adf, model = \
            self.prep_data(
                __mode__=self._EVAL_MODE,
                verbose=verbose,
                *args, **kwargs)

        label_var = \
            self.params.data.label.var \
            if self.params.data.label._int_var is None \
            else self.params.data.label._int_var

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
                logger = logging.getLogger(self._logger_name)
                logger.setLevel(logging.DEBUG)
                logger.addHandler(STDOUT_HANDLER)

                logger.debug(
                    msg='*** EVAL: CONDITION ROBUST TO OUTLIER LABELS: {} {} ***\n'
                        .format(label_var, _outlier_robust_condition))

        # score
        _adf_alias = adf.alias
        assert _adf_alias

        adf = \
            self.score(
                __mode__=self._EVAL_MODE,
                __adf__=
                    adf.filter(
                        condition='{} {}'.format(
                            label_var,
                            _outlier_robust_condition))
                    if __excl_outliers__
                    else adf,
                __model__=model,
                __batch_size__=__batch_size__)

        # if raw_score_col is Vector then convert it to Array
        raw_score_col = self.params.model.score.raw_score_col_prefix + self.params.data.label.var

        if adf.type(raw_score_col) == _VECTOR_TYPE:
            adf('_VECTOR_TO_ARRAY({0}) AS {0}'.format(raw_score_col),
                *(col for col in adf.columns
                      if col != raw_score_col),
                inplace=True)

        adf.alias = _adf_alias + self._EVAL_ADF_ALIAS_SUFFIX

        # cache to calculate multiple metrics quickly
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
                    n_classes=n_classes,
                    labels=self.params.data.label.get('_strings'))) \
                if n_classes \
                else metric_class(
                    label_col=label_var,
                    score_col=raw_score_col)

            evaluators.append(evaluator)

            metrics[self._GLOBAL_EVAL_KEY][evaluator.name] = \
                evaluator(adf, *__class_thresholds__) \
                if n_classes and (metric_name != 'Prevalence') \
                else evaluator(adf)

        id_col = self.params.data.id_col

        if id_col in adf.columns:
            id_col_type_is_str = (adf.type(id_col) == _STR_TYPE)

            ids = adf(
                "SELECT \
                    DISTINCT({}) \
                FROM \
                    this".format(id_col),
                inheritCache=False,
                inheritNRows=False) \
                .toPandas()[id_col]

            for id in tqdm.tqdm(ids):
                _per_id_adf = \
                    adf.filter(
                        condition="{} = {}"
                            .format(
                                id_col,
                                "'{}'".format(id))
                                    if id_col_type_is_str
                                    else id) \
                        .drop(id_col)

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


@_docstr_blueprint
class _DLCrossSectSupervisedBlueprintABC(_CrossSectSupervisedBlueprintABC, _DLSupervisedBlueprintABC):
    __metaclass__ = abc.ABCMeta

    __ModelClass__ = KerasModel

    _DEFAULT_PARAMS = \
        copy.deepcopy(
            _CrossSectSupervisedBlueprintABC._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        copy.deepcopy(
            _DLSupervisedBlueprintABC._DEFAULT_PARAMS))

    def score(self, *args, **kwargs):
        # scoring batch size
        __batch_size__ = kwargs.pop('__batch_size__', 1000)

        # verbosity
        verbose = kwargs.pop('verbose', True)

        # check if scoring for eval purposes
        __eval__ = kwargs.pop(self._MODE_ARG, None) == self._EVAL_MODE
        adf = kwargs.pop('__adf__', None)
        model = kwargs.pop('__model__', None)

        if __eval__:
            assert adf and model

        else:
            adf, model = \
                self.prep_data(
                    __mode__=self._SCORE_MODE,
                    verbose=verbose,
                    *args, **kwargs)

        prep_vec_col = self.params.data._prep_vec_col

        model_file_path = \
            os.path.join(
                model.dir,
                self.params.model._persist.file)

        if fs._ON_LINUX_CLUSTER_WITH_HDFS:
            if model_file_path not in self._MODEL_PATHS_ON_SPARK_WORKER_NODES:
                _tmp_local_file_name = \
                    str(uuid.uuid4())

                _tmp_local_file_path = \
                    os.path.join(
                        '/tmp',
                        _tmp_local_file_name)

                shutil.copyfile(
                    src=model_file_path,
                    dst=_tmp_local_file_path)

                arimo.backend.spark.sparkContext.addFile(
                    path=_tmp_local_file_path,
                    recursive=False)

                self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_file_path] = \
                    _tmp_local_file_name   # SparkFiles.get(filename=_tmp_local_file_name)

            _model_file_path = self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_file_path]

        else:
            _model_file_path = model_file_path

        raw_score_col = self.params.model.score.raw_score_col_prefix + self.params.data.label.var

        if __eval__:
            adf(*((self.params.data.label.var
                        if self.params.data.label._int_var is None
                        else self.params.data.label._int_var,
                   prep_vec_col) +
                  ((self.params.data.id_col,)
                   if self.params.data.id_col
                   else ())),
                iCol=self.params.data.id_col,
                tCol=None,
                inplace=True)

        def batch(row_iterator_in_partition):
            rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

            while rows:
                yield \
                    rows, \
                    [numpy.vstack(
                        row[prep_vec_col]
                        for row in rows)],

                rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

        rdd = adf.rdd.mapPartitions(batch)
        
        n_classes = self.params.data.label._n_classes
        binary = (n_classes == 2)

        if isinstance(model._obj, arimo.backend.keras.models.Model):
            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                if cluster:
                    from dl import _load_keras_model
                else:
                    from arimo.util.dl import _load_keras_model

                return [(r +
                         (float(s[0])
                          if (n_classes is None) or binary
                          else s.tolist(),))
                        for r, s in
                        zip(tup[0],
                            _load_keras_model(
                                file_path=_model_file_path)
                            .predict(
                                x=tup[1],
                                batch_size=__batch_size__,
                                verbose=0))]

        score_adf = \
            ADF.create(
                data=rdd.flatMap(score),
                schema=StructType(
                    list(adf.schema) +
                    [StructField(
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
                verifySchema=False,
                iCol=adf.iCol,
                tCol=adf.tCol)

        return score_adf.drop(self.params.data._prep_vec_col)
