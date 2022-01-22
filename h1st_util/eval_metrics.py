import abc
import numpy
import pandas
from sklearn import metrics as skl_metrics

from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
import pyspark.sql
from pyspark.sql import functions

from h1st_util import Namespace
from h1st_util.data_types.spark_sql import _INT_TYPES, _NUM_TYPES, _STR_TYPE


class _MetricABC:
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_pandas_df(self, df, *class_thresholds):
        raise NotImplementedError

    @abc.abstractmethod
    def _eval_spark_df(self, df, *class_thresholds):
        raise NotImplementedError


class _RegrMetricABC(_MetricABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, label_col='y', score_col='__score__'):
        self.label_col = label_col
        self.score_col = score_col

    def __call__(self, df):
        key_tuple = (self.label_col, self.score_col)

        _exists = False

        if hasattr(df, '_eval_metrics'):
            if key_tuple in df._eval_metrics:
                if self.name in df._eval_metrics[key_tuple]:
                    _exists = True

            else:
                df._eval_metrics[key_tuple] = Namespace()

        else:
            df._eval_metrics = {key_tuple: Namespace()}

        if not _exists:
            if isinstance(df, pandas.DataFrame):
                assert df[self.label_col].dtype in (float, int) \
                   and df[self.score_col].dtype in (float, int)

                df._eval_metrics[key_tuple][self.name] = \
                    self._eval_pandas_df(df)

            elif isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame):
                assert ((df.type(self.label_col) in _NUM_TYPES) or df.type(self.label_col).startswith('decimal')) \
                   and ((df.type(self.score_col) in _NUM_TYPES) or df.type(self.score_col).startswith('decimal'))

                df._eval_metrics[key_tuple][self.name] = \
                    self._eval_spark_df(df)

            elif isinstance(df, pyspark.sql.DataFrame) or \
                    isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame):
                label_col_type = df.schema[str(self.label_col)].dataType.simpleString()
                score_col_type = df.schema[str(self.score_col)].dataType.simpleString()

                assert ((label_col_type in _NUM_TYPES) or label_col_type.startswith('decimal')) \
                   and ((score_col_type in _NUM_TYPES) or score_col_type.startswith('decimal'))

                df._eval_metrics[key_tuple][self.name] = \
                    self._eval_spark_df(df)

            else:
                raise ValueError(
                    '*** Type of data to evaluate must be one of '
                    'Pandas.DataFrame, PySpark.SQL.DataFrame or '
                    'inheritor of PySpark.SQL.DataFrame ***')

        return df._eval_metrics[key_tuple][self.name]


class _ClassifMetricABC(_MetricABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self, label_col='y', score_col='__score__', n_classes=2, labels=None):
        self.label_col = label_col
        self.score_col = score_col

        if labels is None:
            assert n_classes >= 2

            self.n_classes = n_classes

            self.labels = tuple(range(n_classes))

        elif n_classes is None:
            assert isinstance(labels, (list, tuple))

            self.n_classes = len(labels)

            assert self.n_classes >= 2

            self.labels = tuple(labels)

        else:
            self.n_classes = n_classes

            self.labels = tuple(labels)

        self.binary = (self.n_classes == 2)

    def __call__(self, df, *class_thresholds):
        if class_thresholds:
            if self.binary:
                if len(class_thresholds) == 1:
                    class_thresholds = (1 - class_thresholds[0], class_thresholds[0])

                else:
                    assert (len(class_thresholds) == 2) \
                       and numpy.allclose(sum(class_thresholds), 1)

            else:
                assert (len(class_thresholds) == self.n_classes) \
                   and numpy.allclose(sum(class_thresholds), 1)

        else:
            class_thresholds = self.n_classes * (1 / self.n_classes,)

        key_tuple = (self.label_col, self.score_col, self.labels, class_thresholds)

        _exists = False

        if hasattr(df, '_eval_metrics'):
            if key_tuple in df._eval_metrics:
                if self.name in df._eval_metrics[key_tuple]:
                    _exists = True

            else:
                df._eval_metrics[key_tuple] = Namespace()

        else:
            df._eval_metrics = {key_tuple: Namespace()}

        if not _exists:
            if isinstance(df, pandas.DataFrame):
                if df[self.label_col].dtype == object:
                    assert self.labels[0] != 0

                df._eval_metrics[key_tuple][self.name] = \
                    self._eval_pandas_df(df, *class_thresholds)

            elif isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame):
                if df.type(self.label_col) == _STR_TYPE:
                    assert self.labels[0] != 0

                df._eval_metrics[key_tuple][self.name] = \
                    self._eval_spark_df(df, *class_thresholds)

            elif isinstance(df, pyspark.sql.DataFrame):
                if df.schema[str(self.label_col)].dataType.simpleString() == _STR_TYPE:
                    assert self.labels[0] != 0

                df._eval_metrics[key_tuple][self.name] = \
                    self._eval_spark_df(df, *class_thresholds)

            else:
                raise ValueError('*** Type of data to evaluate must be one of Pandas.DataFrame, PySpark.SQL.DataFrame or inheritor of PySpark.SQL.DataFrame ***')

        return df._eval_metrics[key_tuple][self.name]


def macro_classif_metric_decor(ClassifMetricClass):
    class C(_ClassifMetricABC):
        name = 'Macro_{0}'.format(ClassifMetricClass.name)

        def _eval_df(self, df, *class_thresholds):
            unweighted_metrics = \
                ClassifMetricClass(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

            return unweighted_metrics.mean(
                    axis='index',
                    skipna=True,
                    level=None) \
                if isinstance(unweighted_metrics, pandas.Series) \
                else unweighted_metrics

        _eval_pandas_df = _eval_spark_df = _eval_df

    return C


def weighted_classif_metric_decor(ClassifMetricClass):
    class C(_ClassifMetricABC):
        name = 'Weighted_{0}'.format(ClassifMetricClass.name)

        def _eval_df(self, df, *class_thresholds):
            unweighted_metrics = \
                ClassifMetricClass(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

            return (Prevalence(
                            label_col=self.label_col,
                            n_classes=self.n_classes,
                            labels=self.labels)
                        (df, *class_thresholds)
                    * unweighted_metrics) \
                   .sum(axis='index',
                        skipna=True,
                        level=None) \
                if isinstance(unweighted_metrics, pandas.Series) \
                else unweighted_metrics

        _eval_pandas_df = _eval_spark_df = _eval_df

    return C


def n(df):
    if hasattr(df, '_N'):
        assert isinstance(df._N, int)

    else:
        df._N = \
            df.nRows \
            if isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame) \
            else (df.count()
                  if isinstance(df, pyspark.sql.DataFrame)
                  else len(df))
        
    return df._N


class MSE(_RegrMetricABC):
    name = 'MSE'

    def _eval_pandas_df(self, df):
        return skl_metrics.mean_squared_error(
                y_true=df[self.label_col],
                y_pred=df[self.score_col],
                sample_weight=None,
                multioutput='uniform_average')

    def _eval_spark_df(self, df):
        return RegressionEvaluator(
                predictionCol=self.score_col,
                labelCol= self.label_col,
                metricName='mse') \
            .evaluate(
                dataset=df)


MeanSquaredError = MeanSqErr = MSE


class RMSE(_RegrMetricABC):
    name = 'RMSE'

    _eval_pandas_df = _eval_spark_df = \
        lambda self, df: MSE(label_col=self.label_col, score_col=self.score_col)(df) ** .5


RootMeanSquaredError = RootMeanSqErr = RMSE


class MAE(_RegrMetricABC):
    name = 'MAE'

    def _eval_pandas_df(self, df):
        return skl_metrics.mean_absolute_error(
                y_true=df[self.label_col],
                y_pred=df[self.score_col],
                sample_weight=None,
                multioutput='uniform_average')

    def _eval_spark_df(self, df):
        return RegressionEvaluator(
                predictionCol=self.score_col,
                labelCol= self.label_col,
                metricName='mae') \
            .evaluate(
                dataset=df)


MeanAbsoluteError = MeanAbsErr = MAE


class MedAE(_RegrMetricABC):
    name = 'MedAE'

    def _eval_pandas_df(self, df):
        return skl_metrics.median_absolute_error(
            y_true=df[self.label_col],
            y_pred=df[self.score_col])

    def _eval_spark_df(self, df):
        _ABS_ERR_COL_NAME = '__AE__'

        spark_df = \
            df._sparkDF \
            if isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame) \
            else df

        return spark_df.select(
            functions.abs(spark_df[self.label_col] - spark_df[self.score_col]).alias(_ABS_ERR_COL_NAME)) \
            .approxQuantile(
                col=_ABS_ERR_COL_NAME,
                probabilities=(.5,),
                relativeError=.0068)[0]


MedianAbsoluteError = MedAbsErr = MedAE


class R2(_RegrMetricABC):
    name = 'R2'

    def _eval_pandas_df(self, df):
        return skl_metrics.r2_score(
                y_true=df[self.label_col],
                y_pred=df[self.score_col],
                sample_weight=None,
                multioutput='uniform_average')

    def _eval_spark_df(self, df):
        return RegressionEvaluator(
                predictionCol=self.score_col,
                labelCol= self.label_col,
                metricName='r2') \
            .evaluate(
                dataset=df)


RSquared = RSquare = RSq = R2


class Prevalence(_ClassifMetricABC):
    name = 'Prevalence'

    def __init__(self, label_col='y', n_classes=2, labels=None):
        super(Prevalence, self).__init__(
            label_col=label_col,
            score_col=NotImplemented,
            n_classes=n_classes,
            labels=labels)

    def __call__(self, df, *class_thresholds):
        _exists = False

        if hasattr(df, '_eval_metrics'):
            if self.label_col in df._eval_metrics:
                if self.name in df._eval_metrics[self.label_col]:
                    _exists = True

            else:
                df._eval_metrics[self.label_col] = Namespace()

        else:
            df._eval_metrics = {self.label_col: Namespace()}

        if not _exists:
            if isinstance(df, pandas.DataFrame):
                if df[self.label_col].dtype == object:
                    assert self.labels[0] != 0

                df._eval_metrics[self.label_col][self.name] = \
                    self._eval_pandas_df(df)

            elif isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame):
                if df.type(self.label_col) == _STR_TYPE:
                    assert self.labels[0] != 0

                df._eval_metrics[self.label_col][self.name] = \
                    self._eval_spark_df(df)

            elif isinstance(df, pyspark.sql.DataFrame):
                if df.schema[str(self.label_col)].dataType.simpleString() == _STR_TYPE:
                    assert self.labels[0] != 0
                    
                df._eval_metrics[self.label_col][self.name] = \
                    self._eval_spark_df(df)

            else:
                raise ValueError('*** Type of data to evaluate must be one of Pandas.DataFrame, PySpark.SQL.DataFrame or inheritor of PySpark.SQL.DataFrame ***')

        return df._eval_metrics[self.label_col][self.name]

    def _eval_pandas_df(self, df):
        prevalences = \
            df.groupby(
                by=self.label_col,
                axis='index',
                level=None,
                as_index=True,
                sort=False,
                group_keys=True,
                squeeze=False) \
            [self.label_col] \
            .count() \
            / n(df)

        assert numpy.allclose(prevalences.sum(), 1)

        return pandas.Series(
            index=self.labels,
            data=((prevalences[label]
                   if label in prevalences.index
                   else 0)
                  for label in
                    (range(self.n_classes)
                     if prevalences.index.dtype in (int, float)
                     else self.labels)))
    
    def _eval_spark_df(self, df):
        _COUNT_COL = '__count__'

        prevalences = \
            df.groupby(self.label_col) \
                .agg(pyspark.sql.functions.count(self.label_col)
                     .alias(_COUNT_COL)) \
                .toPandas() \
                .set_index(
                    self.label_col,
                    drop=True,
                    append=False,
                    inplace=False,
                    verify_integrity=True) \
                [_COUNT_COL] \
            / n(df)

        assert numpy.allclose(prevalences.sum(), 1)

        return pandas.Series(
            index=self.labels,
            data=((prevalences[label]
                   if label in prevalences.index
                   else 0)
                  for label in
                    (range(self.n_classes)
                     if prevalences.index.dtype in (int, float)
                     else self.labels)))


class ConfMat(_ClassifMetricABC):
    name = 'ConfMat'

    def _eval_pandas_df(self, df, *class_thresholds):
        _labels = \
            range(self.n_classes) \
            if df[self.label_col].dtype in (int, float) \
            else self.labels

        conf_mat = \
            skl_metrics.confusion_matrix(
                y_true=df[self.label_col],
                y_pred=df[self.score_col].map(
                    (lambda prob:
                        _labels[prob >= class_thresholds[1]])
                    if self.binary
                    else (lambda probs:
                            _labels[max(range(self.n_classes),
                                        key=lambda i: probs[i] / class_thresholds[i])])),
                labels=_labels,
                sample_weight=None) \
            / n(df)

        assert numpy.allclose(conf_mat.sum(), 1)

        return pandas.DataFrame(
            index=self.labels,
            columns=self.labels,
            data=conf_mat)

    def _eval_spark_df(self, df, *class_thresholds):
        return self._eval_pandas_df(df[self.label_col, self.score_col].toPandas(), *class_thresholds)


ConfusionMatrix = ConfMat


class Acc(_ClassifMetricABC):
    name = 'Acc'

    def _eval_df(self, df, *class_thresholds):
        return numpy.diag(
            ConfMat(label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels)
                (df, *class_thresholds)).sum()

    _eval_pandas_df = _eval_spark_df = _eval_df


Accuracy = Acc


class _PosProportion(_ClassifMetricABC):
    name = '_PosProportion'

    def _eval_df(self, df, *class_thresholds):
        return ConfMat(
                label_col=self.label_col,
                score_col=self.score_col,
                n_classes=self.n_classes,
                labels=self.labels) \
            (df, *class_thresholds) \
            .sum(axis='columns',
                 skipna=True,
                 level=None,
                 numeric_only=True)

    _eval_pandas_df = _eval_spark_df = _eval_df


class _NegProportion(_ClassifMetricABC):
    name = '_NegProportion'

    def _eval_df(self, df, *class_thresholds):
        return 1 - \
            _PosProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


class _PredPosProportion(_ClassifMetricABC):
    name = '_PredPosProportion'

    def _eval_df(self, df, *class_thresholds):
        return ConfMat(
                label_col=self.label_col,
                score_col=self.score_col,
                n_classes=self.n_classes,
                labels=self.labels) \
            (df, *class_thresholds) \
            .sum(axis='index',
                 skipna=True,
                 level=None,
                 numeric_only=True)

    _eval_pandas_df = _eval_spark_df = _eval_df


class _PredNegProportion(_ClassifMetricABC):
    name = '_PredNegProportion'

    def _eval_df(self, df, *class_thresholds):
        return 1 - \
            _PredPosProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


class TruePos(_ClassifMetricABC):
    name = 'TruePos'

    def _eval_df(self, df, *class_thresholds):
        return pandas.Series(
            index=self.labels,
            data=numpy.diag(
                ConfMat(label_col=self.label_col,
                        score_col=self.score_col,
                        n_classes=self.n_classes,
                        labels=self.labels)
                    (df, *class_thresholds)))

    _eval_pandas_df = _eval_spark_df = _eval_df


TruePositive = TPos = TP = Hit = TruePos


class FalsePos(_ClassifMetricABC):
    name = 'FalsePos'

    def _eval_df(self, df, *class_thresholds):
        return _PredPosProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            - TruePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


FalsePositive = FPos = FP = TypeIError = TypeIErr = Type1Error = Type1Err = FalseAlarm = FalsePos


class FalseNeg(_ClassifMetricABC):
    name = 'FalseNeg'

    def _eval_df(self, df, *class_thresholds):
        return _PosProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            - TruePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


FalseNegative = FNeg = FN = TypeIIError = TypeIIErr = Type2Error = Type2Err = Miss = FalseNeg


class TrueNeg(_ClassifMetricABC):
    name = 'TrueNeg'

    def _eval_df(self, df, *class_thresholds):
        return 1 \
            - TruePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            - FalsePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            - FalseNeg(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


TrueNegative = TNeg = TN = TrueNeg


class Precision(_ClassifMetricABC):
    name = 'Precision'

    def _eval_df(self, df, *class_thresholds):
        return TruePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            / _PredPosProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


PositivePredictiveValue = PosPredVal = PPV = Precision

MacroPositivePredictiveValue = MacroPosPredVal = MacroPPV = MacroPrecision = \
    macro_classif_metric_decor(Precision)

WeightedPositivePredictiveValue = WeightedPosPredVal = WeightedPPV = WeightedPrecision = \
    weighted_classif_metric_decor(Precision)


class FalseDiscRate(_ClassifMetricABC):
    name = 'FalseDiscRate'

    def _eval_df(self, df, *class_thresholds):
        return 1 - \
            Precision(label_col=self.label_col,
                      score_col=self.score_col,
                      n_classes=self.n_classes,
                      labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


FalseDiscoveryRate = FDR = FalseDiscRate

MacroFalseDiscoveryRate = MacroFDR = MacroFalseDiscRate = \
    macro_classif_metric_decor(FalseDiscRate)

WeightedFalseDiscoveryRate = WeightedFDR = WeightedFalseDiscRate = \
    weighted_classif_metric_decor(FalseDiscRate)


class FalseOmissionRate(_ClassifMetricABC):
    name = 'FalseOmissionRate'

    def _eval_df(self, df, *class_thresholds):
        return FalseNeg(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            / _PredNegProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


FOR = FalseOmissionRate

MacroFOR = MacroFalseOmissionRate = \
    macro_classif_metric_decor(FalseOmissionRate)

WeightedFOR = WeightedFalseOmissionRate = \
    weighted_classif_metric_decor(FalseOmissionRate)


class NegPredVal(_ClassifMetricABC):
    name = 'NegPredVal'

    def _eval_df(self, df, *class_thresholds):
        return 1 - \
            FalseOmissionRate(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


NegativePredictiveValue = NPV = NegPredVal

MacroNegativePredictiveValue = MacroNPV = MacroNegPredVal = \
    macro_classif_metric_decor(NegPredVal)

WeightedNegativePredictiveValue = WeightedNPV = WeightedNegPredVal = \
    weighted_classif_metric_decor(NegPredVal)


class Recall(_ClassifMetricABC):
    name = 'Recall'

    def _eval_df(self, df, *class_thresholds):
        return TruePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            / _PosProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


TruePositiveRate = TruePosRate = TPR = Sensitivity = DetectionProbability = DetectProb = HitRate = Recall

MacroTruePositiveRate = MacroTruePosRate = MacroTPR = MacroSensitivity = \
    MacroDetectionProbability = MacroDetectProb = MacroHitRate = MacroRecall = \
    macro_classif_metric_decor(Recall)

WeightedTruePositiveRate = WeightedTruePosRate = WeightedTPR = WeightedSensitivity = \
    WeightedDetectionProbability = WeightedDetectProb = WeightedHitRate = WeightedRecall = \
    weighted_classif_metric_decor(Recall)


class FalsePosRate(_ClassifMetricABC):
    name = 'FalsePosRate'

    def _eval_df(self, df, *class_thresholds):
        return FalsePos(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            / _NegProportion(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


FalsePositiveRate = FPR = FallOut = Fallout = FalseAlarmProbability = FalseAlarmProb = FalsePosRate

MacroFalsePositiveRate = MacroFPR = MacroFallOut = MacroFallout = \
    MacroFalseAlarmProbability = MacroFalseAlarmProb = MacroFalsePosRate = \
    macro_classif_metric_decor(FalsePosRate)

WeightedFalsePositiveRate = WeightedFPR = WeightedFallOut = WeightedFallout = \
    WeightedFalseAlarmProbability = WeightedFalseAlarmProb = WeightedFalsePosRate = \
    weighted_classif_metric_decor(FalsePosRate)


class FalseNegRate(_ClassifMetricABC):
    name = 'FalseNegRate'

    def _eval_df(self, df, *class_thresholds):
        return 1 - \
            Recall(label_col=self.label_col,
                   score_col=self.score_col,
                   n_classes=self.n_classes,
                   labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


FalseNegativeRate = FNR = MissRate = FalseNegRate

MacroFalseNegativeRate = MacroFNR = MacroMissRate = MacroFalseNegRate = \
    macro_classif_metric_decor(FalseNegRate)

WeightedFalseNegativeRate = WeightedFNR = WeightedMissRate = WeightedFalseNegRate = \
    weighted_classif_metric_decor(FalseNegRate)


class Specificity(_ClassifMetricABC):
    name = 'Specificity'

    def _eval_df(self, df, *class_thresholds):
        return 1 - \
            FalsePosRate(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


TrueNegativeRate = TrueNegRate = TNR = SPC = Specificity

MacroTrueNegativeRate = MacroTrueNegRate = MacroTNR = MacroSPC = MacroSpecificity = \
    macro_classif_metric_decor(Specificity)

WeightedTrueNegativeRate = WeightedTrueNegRate = WeightedTNR = WeightedSPC = WeightedSpecificity = \
    weighted_classif_metric_decor(Specificity)


class PosLikelihoodRatio(_ClassifMetricABC):
    name = 'PosLikelihoodRatio'

    def _eval_df(self, df, *class_thresholds):
        return TPR(label_col=self.label_col,
                   score_col=self.score_col,
                   n_classes=self.n_classes,
                   labels=self.labels) \
                (df, *class_thresholds) \
            / FPR(label_col=self.label_col,
                  score_col=self.score_col,
                  n_classes=self.n_classes,
                  labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


PositiveLikelihoodRatio = PosLR = PosLikelihoodRatio

MacroPositiveLikelihoodRatio = MacroPosLR = MacroPosLikelihoodRatio = \
    macro_classif_metric_decor(PosLikelihoodRatio)

WeightedPositiveLikelihoodRatio = WeightedPosLR = WeightedPosLikelihoodRatio = \
    weighted_classif_metric_decor(PosLikelihoodRatio)


class NegLikelihoodRatio(_ClassifMetricABC):
    name = 'NegLikelihoodRatio'

    def _eval_df(self, df, *class_thresholds):
        return FNR(label_col=self.label_col,
                   score_col=self.score_col,
                   n_classes=self.n_classes,
                   labels=self.labels) \
                (df, *class_thresholds) \
            / TNR(label_col=self.label_col,
                  score_col=self.score_col,
                  n_classes=self.n_classes,
                  labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


NegativeLikelihoodRatio = NegLR = NegLikelihoodRatio

MacroNegativeLikelihoodRatio = MacroNegLR = MacroNegLikelihoodRatio = \
    macro_classif_metric_decor(NegLikelihoodRatio)

WeightedNegativeLikelihoodRatio = WeightedNegLR = WeightedNegLikelihoodRatio = \
    weighted_classif_metric_decor(NegLikelihoodRatio)


class DiagnOddsRatio(_ClassifMetricABC):
    name = 'DiagnOddsRatio'

    def _eval_df(self, df, *class_thresholds):
        return PosLR(label_col=self.label_col,
                     score_col=self.score_col,
                     n_classes=self.n_classes,
                     labels=self.labels) \
                (df, *class_thresholds) \
            / NegLR(label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


DiagnosticOddsRatio = DOR = DiagnOddsRatio

MacroDiagnosticOddsRatio = MacroDOR = MacroDiagnOddsRatio = \
    macro_classif_metric_decor(DiagnOddsRatio)

WeightedDiagnosticOddsRatio = WeightedDOR = WeightedDiagnOddsRatio = \
    weighted_classif_metric_decor(DiagnOddsRatio)


class F1(_ClassifMetricABC):
    name = 'F1'

    def _eval_df(self, df, *class_thresholds):
        precision = \
            Precision(
                label_col=self.label_col,
                score_col=self.score_col,
                n_classes=self.n_classes,
                labels=self.labels) \
            (df, *class_thresholds)

        recall = \
            Recall(
                label_col=self.label_col,
                score_col=self.score_col,
                n_classes=self.n_classes,
                labels=self.labels) \
            (df, *class_thresholds)

        return (2 * precision * recall) / (precision + recall)

    _eval_pandas_df = _eval_spark_df = _eval_df


BalancedFScore = BalFScore = F1Score = FMeasure = F1

MacroBalancedFScore = MacroBalFScore = MacroF1Score = MacroFMeasure = MacroF1 = \
    macro_classif_metric_decor(F1)

WeightedBalancedFScore = WeightedBalFScore = WeightedF1Score = WeightedFMeasure = WeightedF1 = \
    weighted_classif_metric_decor(F1)


class GMeasure(_ClassifMetricABC):
    name = 'GMeasure'

    def _eval_df(self, df, *class_thresholds):
        precision = \
            Precision(
                label_col=self.label_col,
                score_col=self.score_col,
                n_classes=self.n_classes,
                labels=self.labels) \
            (df, *class_thresholds)

        recall = \
            Recall(
                label_col=self.label_col,
                score_col=self.score_col,
                n_classes=self.n_classes,
                labels=self.labels) \
            (df, *class_thresholds)

        return (precision * recall) ** .5

    _eval_pandas_df = _eval_spark_df = _eval_df


FowlkesMallowsIndex = GMeasure

MacroFowlkesMallowsIndex = MacroGMeasure = \
    macro_classif_metric_decor(GMeasure)

WeightedFowlkesMallowsIndex = WeightedGMeasure = \
    weighted_classif_metric_decor(GMeasure)


class Informedness(_ClassifMetricABC):
    name = 'Informedness'

    def _eval_df(self, df, *class_thresholds):
        return -1 \
            + Sensitivity(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds) \
            + Specificity(
                    label_col=self.label_col,
                    score_col=self.score_col,
                    n_classes=self.n_classes,
                    labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


YoudenJStatistic = YoudenJStat = YoudenJ = YoudenIndex = BookmakerInformedness = Informedness

MacroYoudenJStatistic = MacroYoudenJStat = MacroYoudenJ = MacroYoudenIndex = \
    MacroBookmakerInformedness = MacroInformedness = \
    macro_classif_metric_decor(Informedness)

WeightedYoudenJStatistic = WeightedYoudenJStat = WeightedYoudenJ = WeightedYoudenIndex = \
    WeightedBookmakerInformedness = WeightedInformedness = \
    weighted_classif_metric_decor(Informedness)


class Markedness(_ClassifMetricABC):
    name = 'Markedness'

    def _eval_df(self, df, *class_thresholds):
        return -1 \
            + PPV(label_col=self.label_col,
                  score_col=self.score_col,
                  n_classes=self.n_classes,
                  labels=self.labels) \
                (df, *class_thresholds) \
            + NPV(label_col=self.label_col,
                  score_col=self.score_col,
                  n_classes=self.n_classes,
                  labels=self.labels) \
                (df, *class_thresholds)

    _eval_pandas_df = _eval_spark_df = _eval_df


MacroMarkedness = \
    macro_classif_metric_decor(Markedness)

WeightedMarkedness = \
    weighted_classif_metric_decor(Markedness)


class PR_AuC(_ClassifMetricABC):
    name = 'PR_AuC'

    def _eval_pandas_df(self, df, *class_thresholds):
        _int_label_type = (df[self.label_col].dtype in (int, float))

        return pandas.Series(
            index=self.labels,
            data=(skl_metrics.average_precision_score(
                    y_true=
                        (df[self.label_col]
                         if i
                         else (1 - df[self.label_col]))
                        if self.binary and _int_label_type
                        else (df[self.label_col] == label),
                    y_score=
                        (df[self.score_col]
                         if i
                         else (1 - df[self.score_col]))
                        if self.binary
                        else [probs[i]
                              for _, probs in df[self.score_col].iteritems()],
                    average=None,
                    sample_weight=None)
                  for i, label in
                    enumerate(range(self.n_classes)
                              if _int_label_type
                              else self.labels)))

    def _eval_spark_df(self, df, *class_thresholds):
        _int_label_type = \
            ((df.type(self.label_col)
              if isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame)
              else df.schema[str(self.label_col)].dataType.simpleString()) in _INT_TYPES)

        evaluator = \
            BinaryClassificationEvaluator(
                rawPredictionCol=self.score_col,
                labelCol=self.label_col,
                metricName='areaUnderPR')

        prevalences = \
            Prevalence(
                label_col=self.label_col,
                n_classes=self.n_classes,
                labels=self.labels)(df)

        return pandas.Series(
            index=self.labels,
            data=((evaluator.evaluate(
                        dataset=df.selectExpr(
                            (self.label_col
                                    if i
                                    else '(1 - {0}) AS {0}'.format(self.label_col))
                                if self.binary and _int_label_type
                                else 'IF({0} = {1}, 1, 0) AS {0}'.format(
                                        self.label_col,
                                        label if _int_label_type
                                              else "'{0}'".format(label)),
                            (self.score_col
                                    if i
                                    else '(1 - {0}) AS {0}'.format(self.score_col))
                                if self.binary
                                else '{0}[{1}] AS {0}'.format(self.score_col, i)))
                   if prevalences.iloc[i]
                   else numpy.nan)
                  for i, label in
                    enumerate(range(self.n_classes)
                              if _int_label_type
                              else self.labels)))


AreaUnderPRCurve = AreaUnderPR = AveragePrecision = AvgPrecision = PR_AuC

MacroAreaUnderPRCurve = MacroAreaUnderPR = MacroAveragePrecision = MacroAvgPrecision = Macro_PR_AuC = \
    macro_classif_metric_decor(PR_AuC)

WeightedAreaUnderPRCurve = WeightedAreaUnderPR = WeightedAveragePrecision = WeightedAvgPrecision = Weighted_PR_AuC = \
    weighted_classif_metric_decor(PR_AuC)


class ROC_AuC(_ClassifMetricABC):
    name = 'ROC_AuC'

    def _eval_pandas_df(self, df, *class_thresholds):
        _int_label_type = (df[self.label_col].dtype in (int, float))

        prevalences = \
            Prevalence(
                label_col=self.label_col,
                n_classes=self.n_classes,
                labels=self.labels)(df)

        roc_auc = \
            pandas.Series(
                index=self.labels,
                data=((skl_metrics.roc_auc_score(
                            y_true=
                                (df[self.label_col]
                                 if i
                                 else (1 - df[self.label_col]))
                                if self.binary and _int_label_type
                                else (df[self.label_col] == label),
                            y_score=
                                (df[self.score_col]
                                 if i
                                 else (1 - df[self.score_col]))
                                if self.binary
                                else [probs[i]
                                      for _, probs in df[self.score_col].iteritems()],
                            average=None,
                            sample_weight=None)
                       if prevalences.iloc[i]
                       else numpy.nan)
                      for i, label in
                        enumerate(range(self.n_classes)
                                  if _int_label_type
                                  else self.labels)))

        if self.binary:
            assert numpy.allclose(
                roc_auc.iloc[0],
                roc_auc.iloc[1],
                equal_nan=False)

            return roc_auc.iloc[0]

        else:
            return roc_auc

    def _eval_spark_df(self, df, *class_thresholds):
        _int_label_type = \
            ((df.type(self.label_col)
             if isinstance(getattr(df, '_sparkDF', None), pyspark.sql.DataFrame)
             else df.schema[str(self.label_col)].dataType.simpleString()) in _INT_TYPES)

        evaluator = \
            BinaryClassificationEvaluator(
                rawPredictionCol=self.score_col,
                labelCol=self.label_col,
                metricName='areaUnderROC')

        prevalences = \
            Prevalence(
                label_col=self.label_col,
                n_classes=self.n_classes,
                labels=self.labels)(df)

        roc_auc = \
            pandas.Series(
                index=self.labels,
                data=((evaluator.evaluate(
                            dataset=df.selectExpr(
                                (self.label_col
                                        if i
                                        else '(1 - {0}) AS {0}'.format(self.label_col))
                                    if self.binary and _int_label_type
                                    else 'IF({0} = {1}, 1, 0) AS {0}'.format(
                                            self.label_col,
                                            label if _int_label_type
                                                  else "'{0}'".format(label)),
                                (self.score_col
                                        if i
                                        else '(1 - {0}) AS {0}'.format(self.score_col))
                                    if self.binary
                                    else '{0}[{1}] AS {0}'.format(self.score_col, i)))
                       if prevalences.iloc[i]
                       else numpy.nan)
                      for i, label in
                        enumerate(range(self.n_classes)
                                  if _int_label_type
                                  else self.labels)))

        if self.binary:
            assert numpy.allclose(
                roc_auc.iloc[0],
                roc_auc.iloc[1],
                equal_nan=False)

            return roc_auc.iloc[0]

        else:
            return roc_auc


AreaUnderROCCurve = AreaUnderROC = ROC_AuC

MacroAreaUnderROCCurve = MacroAreaUnderROC = Macro_ROC_AuC = \
    macro_classif_metric_decor(ROC_AuC)

WeightedAreaUnderROCCurve = WeightedAreaUnderROC = Weighted_ROC_AuC = \
    weighted_classif_metric_decor(ROC_AuC)


class _BalAcc(_ClassifMetricABC):
    name = '_BalAcc'

    def _eval_df(self, df, *class_thresholds):
        bal_acc = \
            (TPR(label_col=self.label_col,
                 score_col=self.score_col,
                 n_classes=self.n_classes,
                 labels=self.labels)
                (df, *class_thresholds) +
             TNR(label_col=self.label_col,
                 score_col=self.score_col,
                 n_classes=self.n_classes,
                 labels=self.labels)
                (df, *class_thresholds)) / 2

        if self.binary:
            assert numpy.allclose(
                bal_acc.iloc[0],
                bal_acc.iloc[1],
                equal_nan=False)

            return bal_acc.iloc[0]

        else:
            return bal_acc

    _eval_pandas_df = _eval_spark_df = _eval_df


def _uncache_eval_metrics(df):
    df._eval_metrics = {}
    return df
