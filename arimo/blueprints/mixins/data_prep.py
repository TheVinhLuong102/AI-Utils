from __future__ import division

import abc
import os
import pandas
import uuid

import six
_STR_CLASSES = \
    (str, unicode) \
    if six.PY2 \
    else str

from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
import pyspark.sql

import arimo.backend
from arimo.df.from_files import ArrowADF
from arimo.df.spark import SparkADF
from arimo.df.spark_from_files import ArrowSparkADF
import arimo.eval.metrics
from arimo.util import clean_uuid, Namespace
from arimo.util.iterables import to_iterable
from arimo.util.types.spark_sql import _BOOL_TYPE, _FLOAT_TYPES, _INT_TYPES, _NUM_TYPES, _STR_TYPE
import arimo.debug

from .eval import ClassifEvalMixIn, RegrEvalMixIn


class _DataPrepMixInABC(object):
    __metaclass__ = abc.ABCMeta

    _PREP_ADF_ALIAS_SUFFIX = '__Prep'

    @abc.abstractmethod
    def prep_data(self, df, **kwargs):
        raise NotImplementedError


class _CrossSectDataPrepMixInABC(_DataPrepMixInABC):
    pass


class _TimeSerDataPrepMixInABC(_DataPrepMixInABC):
    pass


class LabeledDataPrepMixIn(_DataPrepMixInABC):
    _INT_LABEL_COL = '__INT_LABEL__'
    _LABELED_ADF_ALIAS_SUFFIX = '__Labeled'

    def prep_data(
            self, df, __mode__='score',
            __from_ensemble__=False, __from_ppp__=False,
            __ohe_cat__=False, __scale_cat__=True, __vectorize__=True,
            verbose=True, **kwargs):
        # check if training, re-training, scoring or evaluating
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

                else:
                    assert isinstance(df, _STR_CLASSES)

                    adf = (ArrowADF
                           if __train__
                           else ArrowSparkADF)(
                        path=df, **kwargs)

        if __train__:
            assert isinstance(adf, ArrowADF)
            
        else:
            assert isinstance(adf, SparkADF)

            if __from_ensemble__ or __from_ppp__:
                    assert adf.alias

            else:
                adf_uuid = clean_uuid(uuid.uuid4())

                adf.alias = \
                    '{}__{}__{}'.format(
                        self.params._uuid,
                        __mode__,
                        adf_uuid)

        if __train__ or __eval__:
            assert self._INT_LABEL_COL not in adf.columns

            # then convert target variable column if necessary
            label_col_type = adf.type(self.params.data.label.var)

            _spark_model = ('spark.ml' in self.params.model.factory.name)

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

                    if _spark_model:
                        self.params.model.factory.labelCol = self._INT_LABEL_COL

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

                    if _spark_model:
                        self.params.model.factory.labelCol = self._INT_LABEL_COL

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

                if _spark_model:
                    self.params.model.factory.labelCol = self.params.data.label.var

            elif _spark_model and __first_train__:
                self.params.model.factory.labelCol = self.params.data.label.var

            if _spark_model:
                self.params.model.factory.featuresCol = self.params.data._prep_vec_col

                if self.params.model.factory.name in \
                        {'pyspark.ml.classification.LogisticRegression',
                         'pyspark.ml.classification.RandomForestClassifier'}:
                    self.params.model.factory.probabilityCol = \
                        self.params.model.score.raw_score_col_prefix + self.params.data.label.var

                elif self.params.model.factory.name in \
                        {'pyspark.ml.classification.GBTClassifier',
                         'pyspark.ml.regression.GBTRegressor',
                         'pyspark.ml.regression.LinearRegression',
                         'pyspark.ml.regression.RandomForestRegressor',
                         'pyspark.ml.regression.GeneralizedLinearRegression'}:
                    self.params.model.factory.predictionCol = \
                        self.params.model.score.raw_score_col_prefix + self.params.data.label.var

            if __train__ and (label_col_type.startswith('decimal') or (label_col_type in _NUM_TYPES)) \
                    and isinstance(self, RegrEvalMixIn) and self.params.data.label.excl_outliers:
                assert self.params.data.label.outlier_tails \
                   and self.params.data.label.outlier_tail_proportion \
                   and (self.params.data.label.outlier_tail_proportion < .5)

                if __first_train__:
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

                        else:
                            self.params.data.label.lower_outlier_threshold = \
                                adf.quantile(
                                    self.params.data.label.var,
                                    q=self.params.data.label.outlier_tail_proportion,
                                    relativeError=self.params.data.label.outlier_tail_proportion / 3)

                    elif _calc_upper_outlier_threshold:
                        self.params.data.label.upper_outlier_threshold = \
                            adf.quantile(
                                self.params.data.label.var,
                                q=1 - self.params.data.label.outlier_tail_proportion,
                                relativeError=self.params.data.label.outlier_tail_proportion / 3)

                _lower_outlier_threshold_applicable = \
                    pandas.notnull(self.params.data.label.lower_outlier_threshold)

                _upper_outlier_threshold_applicable = \
                    pandas.notnull(self.params.data.label.upper_outlier_threshold)

                if _lower_outlier_threshold_applicable or _upper_outlier_threshold_applicable:
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
                        self.stdout_logger.debug(
                            msg='*** DATA PREP FOR TRAIN: CONDITION ROBUST TO LABEL OUTLIERS: {} {}... ***\n'
                                .format(self.params.data.label.var, _outlier_robust_condition))

                    if isinstance(adf, SparkADF):
                        adf('IF({0} {1}, {0}, NULL) AS {0}'
                                .format(
                                    self.params.data.label.var,
                                    _outlier_robust_condition),
                            *(col for col in adf.columns
                                  if col != self.params.data.label.var),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

            if isinstance(adf, SparkADF):
                adf.alias += self._LABELED_ADF_ALIAS_SUFFIX

        if __from_ensemble__ or __from_ppp__:
            if __vectorize__:
                assert self.params.data._prep_vec_col in adf.columns

        else:
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
                            [self.params.data.label.var])

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
                        self._PREP_ADF_ALIAS_SUFFIX))

            if __train__ or __eval__:
                if __first_train__:
                    self.params.data.pred_vars = \
                        tuple(self.params.data.pred_vars
                              .intersection(set(cat_orig_to_prep_col_map)
                                            .union(num_orig_to_prep_col_map)))

                    self.params.data.pred_vars_incl = \
                        self.params.data.pred_vars_excl = None

                    self.params.data._cat_prep_cols_metadata = \
                        dict(cat_prep_col_n_metadata
                             for cat_orig_col, cat_prep_col_n_metadata in cat_orig_to_prep_col_map.items()
                             if cat_orig_col in self.params.data.pred_vars)

                    self.params.data._cat_prep_cols = \
                        tuple(self.params.data._cat_prep_cols_metadata)

                    self.params.data._num_prep_cols_metadata = \
                        dict(num_prep_col_n_metadata
                             for num_orig_col, num_prep_col_n_metadata in num_orig_to_prep_col_map.items()
                             if num_orig_col in self.params.data.pred_vars)

                    self.params.data._num_prep_cols = \
                        tuple(self.params.data._num_prep_cols_metadata)

                    self.params.data._prep_vec_size = \
                        adf._colWidth(self.params.data._prep_vec_col) \
                        if __vectorize__ \
                        else (len(self.params.data._num_prep_cols) +
                              (sum(_cat_prep_col_metadata['NCats']
                                   for _cat_prep_col_metadata in self.params.data._cat_prep_cols_metadata.values())
                               if cat_orig_to_prep_col_map['__OHE__']
                               else len(self.params.data._cat_prep_cols)))

                adf(self.params.data.label.var
                        if self.params.data.label._int_var is None
                        else self.params.data.label._int_var,
                    *((() if self.params.data.id_col in adf.indexCols
                          else (self.params.data.id_col,)) +
                      adf.indexCols + adf.tAuxCols +
                      ((self.params.data._prep_vec_col,)
                       if __vectorize__
                       else (self.params.data._cat_prep_cols + self.params.data._num_prep_cols))),
                    inheritCache=True,
                    inheritNRows=True,
                    inplace=True)

        return adf


class PPPDataPrepMixIn(_DataPrepMixInABC):
    _TO_SCORE_ALL_VARS_ADF_ALIAS_SUFFIX = '__toScoreAllVars'

    def prep_data(self, df, __mode__='score', __vectorize__=True, verbose=True, **kwargs):
        # check if training, re-training, scoring or evaluating
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

            else:
                assert isinstance(df, _STR_CLASSES)

                adf = (ArrowADF
                       if __train__
                       else ArrowSparkADF)(
                    path=df, **kwargs)

        assert (self.params.data.id_col in adf.columns) \
           and (self.params.data.time_col in adf.columns)
        
        if __train__:
            assert isinstance(adf, ArrowADF)

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
            assert isinstance(adf, SparkADF)
            
            adf_uuid = clean_uuid(uuid.uuid4())

            adf.alias = \
                '{}__{}__{}'.format(
                    self.params._uuid,
                    __mode__,
                    adf_uuid)

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

                verbose=verbose,

                alias='{}__{}__{}{}'.format(
                    self.params._uuid,
                    __mode__,
                    adf_uuid,
                    self._PREP_ADF_ALIAS_SUFFIX))

        if __train__:
            component_labeled_adfs = Namespace()

            for label_var_name in set(self.params.model.component_blueprints).intersection(adf.contentCols):
                if adf.suffNonNull(label_var_name):
                    component_blueprint_params = \
                        self.params.model.component_blueprints[label_var_name]

                    if __first_train__:
                        component_blueprint_params.data.pred_vars = \
                            tuple(component_blueprint_params.data.pred_vars
                                  .intersection(set(cat_orig_to_prep_col_map)
                                                .union(num_orig_to_prep_col_map)))

                        component_blueprint_params.data.pred_vars_incl = \
                            component_blueprint_params.data.pred_vars_excl = None

                        component_blueprint_params.data._cat_prep_cols_metadata = \
                            dict(cat_prep_col_n_metadata
                                 for cat_orig_col, cat_prep_col_n_metadata in cat_orig_to_prep_col_map.items()
                                 if cat_orig_col in component_blueprint_params.data.pred_vars)

                        component_blueprint_params.data._cat_prep_cols = \
                            tuple(component_blueprint_params.data._cat_prep_cols_metadata)

                        component_blueprint_params.data._num_prep_cols_metadata = \
                            dict(num_prep_col_n_metadata
                                 for num_orig_col, num_prep_col_n_metadata in num_orig_to_prep_col_map.items()
                                 if num_orig_col in component_blueprint_params.data.pred_vars)

                        component_blueprint_params.data._num_prep_cols = \
                            tuple(component_blueprint_params.data._num_prep_cols_metadata)

                        component_blueprint_params.data._prep_vec_size = \
                            len(component_blueprint_params.data._num_prep_cols) + \
                            (sum(_cat_prep_col_metadata['NCats']
                                 for _cat_prep_col_metadata in component_blueprint_params.data._cat_prep_cols_metadata.values())
                             if cat_orig_to_prep_col_map['__OHE__']
                             else len(component_blueprint_params.data._cat_prep_cols))

                    component_labeled_adfs[label_var_name] = \
                        adf[[label_var_name] +
                            list(adf.indexCols + adf.tAuxCols +
                                 component_blueprint_params.data._cat_prep_cols +
                                 component_blueprint_params.data._num_prep_cols)]

            # save Blueprint & data transforms
            self.save()

            return component_labeled_adfs

        else:
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
                adf.alias += self._TO_SCORE_ALL_VARS_ADF_ALIAS_SUFFIX
                return adf
