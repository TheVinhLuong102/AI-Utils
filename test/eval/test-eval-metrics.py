from __future__ import division, print_function

import numpy
import pandas
from sklearn import metrics as skl_metrics

from arimo.df.spark import SparkADF
from arimo.eval.metrics import \
    MSE, RMSE, MAE, R2, \
    Prevalence, ConfMat, Acc, Precision, Recall, F1, PR_AuC, ROC_AuC, \
    _uncache_eval_metrics


Y_BIN_INT_COL = 'y_bin_int'
Y_BIN_STR_COL = 'y_bin_str'
Y_MULTI_INT_COL = 'y_multi_int'
Y_MULTI_STR_COL = 'y_multi_str'
SCORE_BIN_COL = 'score_bin'
SCORE_MULTI_COL = 'score_multi'

BIN_LABELS = ['non-A', 'A']
MULTI_LABELS = ['o', 'A', 'B', 'p', 'C', 'q']
N_MULTI_CLASSES = len(MULTI_LABELS)

BIN_THRESHOLD = numpy.random.random()
MULTI_THRESHOLDS = numpy.random.dirichlet(numpy.ones(N_MULTI_CLASSES), size=None).tolist()


df = pandas.DataFrame(
    data={
        Y_BIN_INT_COL: 15 * [0] + 3 * [1],
        Y_BIN_STR_COL: 15 * ['non-A'] + 3 * ['A'],
        Y_MULTI_INT_COL: 6 * [2] + 9 * [4] + 3 * [1],
        Y_MULTI_STR_COL: 6 * ['B'] + 9 * ['C'] + 3 * ['A'],
        SCORE_BIN_COL: numpy.random.random(18),
        SCORE_MULTI_COL: [numpy.random.dirichlet(numpy.ones(N_MULTI_CLASSES), size=None).tolist()
                          for _ in range(18)]})
adf = SparkADF.create(data=df)
sdf = adf._sparkDF

n = len(df)

_skl_y_true_bin_int = df[Y_BIN_INT_COL]
_skl_y_true_bin_str = df[Y_BIN_STR_COL]
_skl_y_score_bin = df[SCORE_BIN_COL]
_skl_y_pred_bin_int = (df[SCORE_BIN_COL] >= BIN_THRESHOLD)
_skl_y_pred_bin_str = [BIN_LABELS[v]
                       for _, v in _skl_y_pred_bin_int.iteritems()]

_skl_y_true_multi_int = df[Y_MULTI_INT_COL]
_skl_y_true_multi_str = df[Y_MULTI_STR_COL]
_skl_y_pred_multi_int = [max(range(N_MULTI_CLASSES),
                             key=lambda i: probs[i] / MULTI_THRESHOLDS[i])
                         for _, probs in df[SCORE_MULTI_COL].iteritems()]
_skl_y_pred_multi_str = [MULTI_LABELS[i]
                         for i in _skl_y_pred_multi_int]


print('\nREGRESSION EVALUATION METRICS:')


mse = MSE(
    label_col=Y_BIN_INT_COL,
    score_col=SCORE_BIN_COL)
pandas_mse = mse(_uncache_eval_metrics(df))
spark_mse = mse(_uncache_eval_metrics(adf))
skl_mse = \
    skl_metrics.mean_squared_error(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_score_bin,
        sample_weight=None,
        multioutput='uniform_average')
print('\nMSE:', pandas_mse, spark_mse, skl_mse)
assert numpy.allclose(pandas_mse, spark_mse) \
   and numpy.allclose(spark_mse, skl_mse)


rmse = RMSE(
    label_col=Y_BIN_INT_COL,
    score_col=SCORE_BIN_COL)
pandas_rmse = rmse(_uncache_eval_metrics(df))
spark_rmse = rmse(_uncache_eval_metrics(adf))
print('\nRMSE:', pandas_rmse, spark_rmse)
assert numpy.allclose(pandas_rmse, spark_rmse)


mae = MAE(
    label_col=Y_BIN_INT_COL,
    score_col=SCORE_BIN_COL)
pandas_mae = mae(_uncache_eval_metrics(df))
spark_mae = mae(_uncache_eval_metrics(adf))
skl_mae = \
    skl_metrics.mean_absolute_error(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_score_bin,
        sample_weight=None,
        multioutput='uniform_average')
print('\nMAE:', pandas_mae, spark_mae, skl_mae)
assert numpy.allclose(pandas_mae, spark_mae) \
   and numpy.allclose(spark_mae, skl_mae)


r2 = R2(
    label_col=Y_BIN_INT_COL,
    score_col=SCORE_BIN_COL)
pandas_r2 = r2(_uncache_eval_metrics(df))
spark_r2 = r2(_uncache_eval_metrics(adf))
skl_r2 = \
    skl_metrics.r2_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_score_bin,
        sample_weight=None,
        multioutput='uniform_average')
print('\nR2:', pandas_r2, spark_r2, skl_r2)
assert numpy.allclose(pandas_r2, spark_r2) \
   and numpy.allclose(spark_r2, skl_r2)


print('\n\nCLASSIFICATION EVALUATION METRICS:')


print('\n\nPREVALENCE:')


prevalence___bin_int_labeled = \
    Prevalence(
        label_col=Y_BIN_INT_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_prevalence___bin_int_labeled = \
    prevalence___bin_int_labeled(
        _uncache_eval_metrics(df))

adf_prevalence___bin_int_labeled = \
    prevalence___bin_int_labeled(
        _uncache_eval_metrics(adf))

spark_prevalence___bin_int_labeled = \
    prevalence___bin_int_labeled(
        _uncache_eval_metrics(sdf))

print('\n\n1. Prevalence (bin, int, labeled):\n\n',
      pandas_prevalence___bin_int_labeled, '\n\n',
      adf_prevalence___bin_int_labeled, '\n\n',
      spark_prevalence___bin_int_labeled, sep='')

assert numpy.allclose(
    pandas_prevalence___bin_int_labeled,
    adf_prevalence___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_prevalence___bin_int_labeled,
    spark_prevalence___bin_int_labeled,
    equal_nan=False)


prevalence___bin_int_unlabeled = \
    Prevalence(
        label_col=Y_BIN_INT_COL,
        n_classes=2,
        labels=None)

pandas_prevalence___bin_int_unlabeled = \
    prevalence___bin_int_unlabeled(
        _uncache_eval_metrics(df))

adf_prevalence___bin_int_unlabeled = \
    prevalence___bin_int_unlabeled(
        _uncache_eval_metrics(adf))

spark_prevalence___bin_int_unlabeled = \
    prevalence___bin_int_unlabeled(
        _uncache_eval_metrics(sdf))

print('\n\n2. Prevalence (bin, int, unlabeled):\n\n',
      pandas_prevalence___bin_int_unlabeled, '\n\n',
      adf_prevalence___bin_int_unlabeled, '\n\n',
      spark_prevalence___bin_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_prevalence___bin_int_unlabeled,
    adf_prevalence___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_prevalence___bin_int_unlabeled,
    spark_prevalence___bin_int_unlabeled,
    equal_nan=False)


prevalence___bin_str_labeled = \
    Prevalence(
        label_col=Y_BIN_STR_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_prevalence___bin_str_labeled = \
    prevalence___bin_str_labeled(
        _uncache_eval_metrics(df))

adf_prevalence___bin_str_labeled = \
    prevalence___bin_str_labeled(
        _uncache_eval_metrics(adf))

spark_prevalence___bin_str_labeled = \
    prevalence___bin_str_labeled(
        _uncache_eval_metrics(sdf))

print('\n\n3. Prevalence (bin, str, labeled):\n\n',
      pandas_prevalence___bin_str_labeled, '\n\n',
      adf_prevalence___bin_str_labeled, '\n\n',
      spark_prevalence___bin_str_labeled, sep='')

assert numpy.allclose(
    pandas_prevalence___bin_str_labeled,
    adf_prevalence___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_prevalence___bin_str_labeled,
    spark_prevalence___bin_str_labeled,
    equal_nan=False)


prevalence___multi_int_labeled = \
    Prevalence(
        label_col=Y_MULTI_INT_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_prevalence___multi_int_labeled = \
    prevalence___multi_int_labeled(
        _uncache_eval_metrics(df))

adf_prevalence___multi_int_labeled = \
    prevalence___multi_int_labeled(
        _uncache_eval_metrics(adf))

spark_prevalence___multi_int_labeled = \
    prevalence___multi_int_labeled(
        _uncache_eval_metrics(sdf))

print('\n\n4. Prevalence (multi, int, labeled):\n\n',
      pandas_prevalence___multi_int_labeled, '\n\n',
      adf_prevalence___multi_int_labeled, '\n\n',
      spark_prevalence___multi_int_labeled, sep='')

assert numpy.allclose(
    pandas_prevalence___multi_int_labeled,
    adf_prevalence___multi_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_prevalence___multi_int_labeled,
    spark_prevalence___multi_int_labeled,
    equal_nan=False)


prevalence___multi_int_unlabeled = \
    Prevalence(
        label_col=Y_MULTI_INT_COL,
        n_classes=N_MULTI_CLASSES,
        labels=None)

pandas_prevalence___multi_int_unlabeled = \
    prevalence___multi_int_unlabeled(
        _uncache_eval_metrics(df))

adf_prevalence___multi_int_unlabeled = \
    prevalence___multi_int_unlabeled(
        _uncache_eval_metrics(adf))

spark_prevalence___multi_int_unlabeled = \
    prevalence___multi_int_unlabeled(
        _uncache_eval_metrics(sdf))

print('\n\n5. Prevalence (multi, int, unlabeled):\n\n',
      pandas_prevalence___multi_int_unlabeled, '\n\n',
      adf_prevalence___multi_int_unlabeled, '\n\n',
      spark_prevalence___multi_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_prevalence___multi_int_unlabeled,
    adf_prevalence___multi_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_prevalence___multi_int_unlabeled,
    spark_prevalence___multi_int_unlabeled,
    equal_nan=False)


prevalence___multi_str_labeled = \
    Prevalence(
        label_col=Y_MULTI_STR_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_prevalence___multi_str_labeled = \
    prevalence___multi_str_labeled(
        _uncache_eval_metrics(df))

adf_prevalence___multi_str_labeled = \
    prevalence___multi_str_labeled(
        _uncache_eval_metrics(adf))

spark_prevalence___multi_str_labeled = \
    prevalence___multi_str_labeled(
        _uncache_eval_metrics(sdf))

print('\n\n6. Prevalence (multi, str, labeled):\n\n',
      pandas_prevalence___multi_str_labeled, '\n\n',
      adf_prevalence___multi_str_labeled, '\n\n',
      spark_prevalence___multi_str_labeled, sep='')

assert numpy.allclose(
    pandas_prevalence___multi_str_labeled,
    adf_prevalence___multi_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_prevalence___multi_str_labeled,
    spark_prevalence___multi_str_labeled,
    equal_nan=False)


print('\n\nCONFUSION MATRIX:')


conf_mat___bin_int_labeled = \
    ConfMat(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_conf_mat___bin_int_labeled = \
    conf_mat___bin_int_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_conf_mat___bin_int_labeled = \
    conf_mat___bin_int_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_conf_mat___bin_int_labeled = \
    conf_mat___bin_int_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_conf_mat___bin_int_labeled = \
    skl_metrics.confusion_matrix(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        sample_weight=None) / n

print('\n\n1. Confusion Matrix (bin, int, labeled):\n\n',
      pandas_conf_mat___bin_int_labeled, '\n\n',
      adf_conf_mat___bin_int_labeled, '\n\n',
      spark_conf_mat___bin_int_labeled, '\n\n',
      skl_conf_mat___bin_int_labeled, sep='')

assert numpy.allclose(
    pandas_conf_mat___bin_int_labeled,
    adf_conf_mat___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___bin_int_labeled,
    spark_conf_mat___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___bin_int_labeled,
    skl_conf_mat___bin_int_labeled,
    equal_nan=False)


conf_mat___bin_int_unlabeled = \
    ConfMat(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=2,
        labels=None)

pandas_conf_mat___bin_int_unlabeled = \
    conf_mat___bin_int_unlabeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_conf_mat___bin_int_unlabeled = \
    conf_mat___bin_int_unlabeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_conf_mat___bin_int_unlabeled = \
    conf_mat___bin_int_unlabeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_conf_mat___bin_int_unlabeled = \
    skl_metrics.confusion_matrix(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        sample_weight=None) / n

print('\n\n2. Confusion Matrix (bin, int, unlabeled):\n\n',
      pandas_conf_mat___bin_int_unlabeled, '\n\n',
      adf_conf_mat___bin_int_unlabeled, '\n\n',
      spark_conf_mat___bin_int_unlabeled, '\n\n',
      skl_conf_mat___bin_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_conf_mat___bin_int_unlabeled,
    adf_conf_mat___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___bin_int_unlabeled,
    spark_conf_mat___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___bin_int_unlabeled,
    skl_conf_mat___bin_int_unlabeled,
    equal_nan=False)


conf_mat___bin_str_labeled = \
    ConfMat(
        label_col=Y_BIN_STR_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_conf_mat___bin_str_labeled = \
    conf_mat___bin_str_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_conf_mat___bin_str_labeled = \
    conf_mat___bin_str_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_conf_mat___bin_str_labeled = \
    conf_mat___bin_str_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_conf_mat___bin_str_labeled = \
    skl_metrics.confusion_matrix(
        y_true=_skl_y_true_bin_str,
        y_pred=_skl_y_pred_bin_str,
        labels=BIN_LABELS,
        sample_weight=None) / n

print('\n\n3. Confusion Matrix (bin, int, labeled):\n\n',
      pandas_conf_mat___bin_str_labeled, '\n\n',
      adf_conf_mat___bin_str_labeled, '\n\n',
      spark_conf_mat___bin_str_labeled, '\n\n',
      skl_conf_mat___bin_str_labeled, sep='')

assert numpy.allclose(
    pandas_conf_mat___bin_str_labeled,
    adf_conf_mat___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___bin_str_labeled,
    spark_conf_mat___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___bin_str_labeled,
    skl_conf_mat___bin_str_labeled,
    equal_nan=False)


conf_mat___multi_int_labeled = \
    ConfMat(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_conf_mat___multi_int_labeled = \
    conf_mat___multi_int_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_conf_mat___multi_int_labeled = \
    conf_mat___multi_int_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_conf_mat___multi_int_labeled = \
    conf_mat___multi_int_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_conf_mat___multi_int_labeled = \
    skl_metrics.confusion_matrix(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        sample_weight=None) / n

print('\n\n4. Confusion Matrix (multi, int, labeled):\n\n',
      pandas_conf_mat___multi_int_labeled, '\n\n',
      adf_conf_mat___multi_int_labeled, '\n\n',
      spark_conf_mat___multi_int_labeled, '\n\n',
      skl_conf_mat___multi_int_labeled, sep='')

assert numpy.allclose(
    pandas_conf_mat___multi_int_labeled,
    adf_conf_mat___multi_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___multi_int_labeled,
    spark_conf_mat___multi_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___multi_int_labeled,
    skl_conf_mat___multi_int_labeled,
    equal_nan=False)


conf_mat___multi_int_unlabeled = \
    ConfMat(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=N_MULTI_CLASSES,
        labels=None)

pandas_conf_mat___multi_int_unlabeled = \
    conf_mat___multi_int_unlabeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_conf_mat___multi_int_unlabeled = \
    conf_mat___multi_int_unlabeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_conf_mat___multi_int_unlabeled = \
    conf_mat___multi_int_unlabeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_conf_mat___multi_int_unlabeled = \
    skl_metrics.confusion_matrix(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        sample_weight=None) / n

print('\n\n5. Confusion Matrix (multi, int, unlabeled):\n\n',
      pandas_conf_mat___multi_int_unlabeled, '\n\n',
      adf_conf_mat___multi_int_unlabeled, '\n\n',
      spark_conf_mat___multi_int_unlabeled, '\n\n',
      skl_conf_mat___multi_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_conf_mat___multi_int_unlabeled,
    adf_conf_mat___multi_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___multi_int_unlabeled,
    spark_conf_mat___multi_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___multi_int_unlabeled,
    skl_conf_mat___multi_int_unlabeled,
    equal_nan=False)


conf_mat___multi_str_labeled = \
    ConfMat(
        label_col=Y_MULTI_STR_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_conf_mat___multi_str_labeled = \
    conf_mat___multi_str_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_conf_mat___multi_str_labeled = \
    conf_mat___multi_str_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_conf_mat___multi_str_labeled = \
    conf_mat___multi_str_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_conf_mat___multi_str_labeled = \
    skl_metrics.confusion_matrix(
        y_true=_skl_y_true_multi_str,
        y_pred=_skl_y_pred_multi_str,
        labels=MULTI_LABELS,
        sample_weight=None) / n

print('\n\n6. Confusion Matrix (multi, str, labeled):\n\n',
      pandas_conf_mat___multi_str_labeled, '\n\n',
      adf_conf_mat___multi_str_labeled, '\n\n',
      spark_conf_mat___multi_str_labeled, '\n\n',
      skl_conf_mat___multi_str_labeled, sep='')

assert numpy.allclose(
    pandas_conf_mat___multi_str_labeled,
    adf_conf_mat___multi_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___multi_str_labeled,
    spark_conf_mat___multi_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_conf_mat___multi_str_labeled,
    skl_conf_mat___multi_str_labeled,
    equal_nan=False)


print('\n\nACCURACY:')


acc___bin_int_labeled = \
    Acc(label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_acc___bin_int_labeled = \
    acc___bin_int_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_acc___bin_int_labeled = \
    acc___bin_int_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_acc___bin_int_labeled = \
    acc___bin_int_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_acc___bin_int_labeled = \
    skl_metrics.accuracy_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        normalize=True,
        sample_weight=None)

print('\n1. Accuracy (bin, int, labeled):',
      pandas_acc___bin_int_labeled,
      adf_acc___bin_int_labeled,
      spark_acc___bin_int_labeled,
      skl_acc___bin_int_labeled)

assert numpy.allclose(pandas_acc___bin_int_labeled, adf_acc___bin_int_labeled) \
   and numpy.allclose(pandas_acc___bin_int_labeled, spark_acc___bin_int_labeled) \
   and numpy.allclose(spark_acc___bin_int_labeled, skl_acc___bin_int_labeled)


acc___bin_int_unlabeled = \
    Acc(label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=2,
        labels=None)

pandas_acc___bin_int_unlabeled = \
    acc___bin_int_unlabeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_acc___bin_int_unlabeled = \
    acc___bin_int_unlabeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_acc___bin_int_unlabeled = \
    acc___bin_int_unlabeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_acc___bin_int_unlabeled = \
    skl_metrics.accuracy_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        normalize=True,
        sample_weight=None)

print('\n2. Accuracy (bin, int, unlabeled):',
      pandas_acc___bin_int_unlabeled,
      adf_acc___bin_int_unlabeled,
      spark_acc___bin_int_unlabeled,
      skl_acc___bin_int_unlabeled)

assert numpy.allclose(pandas_acc___bin_int_unlabeled, adf_acc___bin_int_unlabeled) \
   and numpy.allclose(pandas_acc___bin_int_unlabeled, spark_acc___bin_int_unlabeled) \
   and numpy.allclose(spark_acc___bin_int_unlabeled, skl_acc___bin_int_unlabeled)


acc___bin_str_labeled = \
    Acc(label_col=Y_BIN_STR_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_acc___bin_str_labeled = \
    acc___bin_str_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_acc___bin_str_labeled = \
    acc___bin_str_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_acc___bin_str_labeled = \
    acc___bin_str_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_acc___bin_str_labeled = \
    skl_metrics.accuracy_score(
        y_true=_skl_y_true_bin_str,
        y_pred=_skl_y_pred_bin_str,
        normalize=True,
        sample_weight=None)

print('\n3. Accuracy (bin, str, labeled):',
      pandas_acc___bin_str_labeled,
      adf_acc___bin_str_labeled,
      spark_acc___bin_str_labeled,
      skl_acc___bin_str_labeled)

assert numpy.allclose(pandas_acc___bin_str_labeled, adf_acc___bin_str_labeled) \
   and numpy.allclose(pandas_acc___bin_str_labeled, spark_acc___bin_str_labeled) \
   and numpy.allclose(spark_acc___bin_str_labeled, skl_acc___bin_str_labeled)


acc___multi_int_labeled = \
    Acc(label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_acc___multi_int_labeled = \
    acc___multi_int_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_acc___multi_int_labeled = \
    acc___multi_int_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_acc___multi_int_labeled = \
    acc___multi_int_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_acc___multi_int_labeled = \
    skl_metrics.accuracy_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        normalize=True,
        sample_weight=None)

print('\n4. Accuracy (multi, int, labeled):',
      pandas_acc___multi_int_labeled,
      adf_acc___multi_int_labeled,
      spark_acc___multi_int_labeled,
      skl_acc___multi_int_labeled)

assert numpy.allclose(pandas_acc___multi_int_labeled, adf_acc___multi_int_labeled) \
   and numpy.allclose(pandas_acc___multi_int_labeled, spark_acc___multi_int_labeled) \
   and numpy.allclose(spark_acc___multi_int_labeled, skl_acc___multi_int_labeled)


acc___multi_int_unlabeled = \
    Acc(label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=N_MULTI_CLASSES,
        labels=None)

pandas_acc___multi_int_unlabeled = \
    acc___multi_int_unlabeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_acc___multi_int_unlabeled = \
    acc___multi_int_unlabeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_acc___multi_int_unlabeled = \
    acc___multi_int_unlabeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_acc___multi_int_unlabeled = \
    skl_metrics.accuracy_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        normalize=True,
        sample_weight=None)

print('\n5. Accuracy (multi, int, unlabeled):',
      pandas_acc___multi_int_unlabeled,
      adf_acc___multi_int_unlabeled,
      spark_acc___multi_int_unlabeled,
      skl_acc___multi_int_unlabeled)

assert numpy.allclose(pandas_acc___multi_int_unlabeled, adf_acc___multi_int_unlabeled) \
   and numpy.allclose(pandas_acc___multi_int_unlabeled, spark_acc___multi_int_unlabeled) \
   and numpy.allclose(spark_acc___multi_int_unlabeled, skl_acc___multi_int_unlabeled)


acc___multi_str_labeled = \
    Acc(label_col=Y_MULTI_STR_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_acc___multi_str_labeled = \
    acc___multi_str_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_acc___multi_str_labeled = \
    acc___multi_str_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_acc___multi_str_labeled = \
    acc___multi_str_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_acc___multi_str_labeled = \
    skl_metrics.accuracy_score(
        y_true=_skl_y_true_multi_str,
        y_pred=_skl_y_pred_multi_str,
        normalize=True,
        sample_weight=None)

print('\n6. Accuracy (multi, str, labeled):',
      pandas_acc___multi_str_labeled,
      adf_acc___multi_str_labeled,
      spark_acc___multi_str_labeled,
      skl_acc___multi_str_labeled)

assert numpy.allclose(pandas_acc___multi_str_labeled, adf_acc___multi_str_labeled) \
   and numpy.allclose(pandas_acc___multi_str_labeled, spark_acc___multi_str_labeled) \
   and numpy.allclose(spark_acc___multi_str_labeled, skl_acc___multi_str_labeled)


print('\n\nPRECISION:')


precision___bin_int_labeled = \
    Precision(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_precision___bin_int_labeled = \
    precision___bin_int_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_precision___bin_int_labeled = \
    precision___bin_int_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_precision___bin_int_labeled = \
    precision___bin_int_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_precision___bin_int_labeled = \
    skl_metrics.precision_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n1. Precision (bin, int, labeled):\n\n',
      pandas_precision___bin_int_labeled, '\n\n',
      adf_precision___bin_int_labeled, '\n\n',
      spark_precision___bin_int_labeled, '\n\n',
      skl_precision___bin_int_labeled, sep='')

assert numpy.allclose(
    pandas_precision___bin_int_labeled,
    adf_precision___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_precision___bin_int_labeled,
    spark_precision___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_precision___bin_int_labeled],
    skl_precision___bin_int_labeled,
    equal_nan=False)


precision___bin_int_unlabeled = \
    Precision(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=2,
        labels=None)

pandas_precision___bin_int_unlabeled = \
    precision___bin_int_unlabeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_precision___bin_int_unlabeled = \
    precision___bin_int_unlabeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_precision___bin_int_unlabeled = \
    precision___bin_int_unlabeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_precision___bin_int_unlabeled = \
    skl_metrics.precision_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n2. Precision (bin, int, unlabeled):\n\n',
      pandas_precision___bin_int_unlabeled, '\n\n',
      adf_precision___bin_int_unlabeled, '\n\n',
      spark_precision___bin_int_unlabeled, '\n\n',
      skl_precision___bin_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_precision___bin_int_unlabeled,
    adf_precision___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_precision___bin_int_unlabeled,
    spark_precision___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_precision___bin_int_unlabeled],
    skl_precision___bin_int_unlabeled,
    equal_nan=False)


precision___bin_str_labeled = \
    Precision(
        label_col=Y_BIN_STR_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_precision___bin_str_labeled = \
    precision___bin_str_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_precision___bin_str_labeled = \
    precision___bin_str_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_precision___bin_str_labeled = \
    precision___bin_str_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_precision___bin_str_labeled = \
    skl_metrics.precision_score(
        y_true=_skl_y_true_bin_str,
        y_pred=_skl_y_pred_bin_str,
        labels=BIN_LABELS,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n3. Precision (bin, int, labeled):\n\n',
      pandas_precision___bin_str_labeled, '\n\n',
      adf_precision___bin_str_labeled, '\n\n',
      spark_precision___bin_str_labeled, '\n\n',
      skl_precision___bin_str_labeled, sep='')

assert numpy.allclose(
    pandas_precision___bin_str_labeled,
    adf_precision___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_precision___bin_str_labeled,
    spark_precision___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_precision___bin_str_labeled],
    skl_precision___bin_str_labeled,
    equal_nan=False)


precision___multi_int_labeled = \
    Precision(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_precision___multi_int_labeled = \
    precision___multi_int_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_precision___multi_int_labeled = \
    precision___multi_int_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_precision___multi_int_labeled = \
    precision___multi_int_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_precision___multi_int_labeled = \
    skl_metrics.precision_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n4. Precision (multi, int, labeled):\n\n',
      pandas_precision___multi_int_labeled, '\n\n',
      adf_precision___multi_int_labeled, '\n\n',
      spark_precision___multi_int_labeled, '\n\n',
      skl_precision___multi_int_labeled, sep='')

assert numpy.allclose(
    pandas_precision___multi_int_labeled,
    adf_precision___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_precision___multi_int_labeled,
    spark_precision___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_precision___multi_int_labeled],
    skl_precision___multi_int_labeled,
    equal_nan=False)


precision___multi_int_unlabeled = \
    Precision(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=N_MULTI_CLASSES,
        labels=None)

pandas_precision___multi_int_unlabeled = \
    precision___multi_int_unlabeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_precision___multi_int_unlabeled = \
    precision___multi_int_unlabeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_precision___multi_int_unlabeled = \
    precision___multi_int_unlabeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_precision___multi_int_unlabeled = \
    skl_metrics.precision_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n5. Precision (multi, int, unlabeled):\n\n',
      pandas_precision___multi_int_unlabeled, '\n\n',
      adf_precision___multi_int_unlabeled, '\n\n',
      spark_precision___multi_int_unlabeled, '\n\n',
      skl_precision___multi_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_precision___multi_int_unlabeled,
    adf_precision___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_precision___multi_int_unlabeled,
    spark_precision___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_precision___multi_int_unlabeled],
    skl_precision___multi_int_unlabeled,
    equal_nan=False)


precision___multi_str_labeled = \
    Precision(
        label_col=Y_MULTI_STR_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_precision___multi_str_labeled = \
    precision___multi_str_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_precision___multi_str_labeled = \
    precision___multi_str_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_precision___multi_str_labeled = \
    precision___multi_str_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_precision___multi_str_labeled = \
    skl_metrics.precision_score(
        y_true=_skl_y_true_multi_str,
        y_pred=_skl_y_pred_multi_str,
        labels=MULTI_LABELS,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n6. Precision (multi, str, labeled):\n\n',
      pandas_precision___multi_str_labeled, '\n\n',
      adf_precision___multi_str_labeled, '\n\n',
      spark_precision___multi_str_labeled, '\n\n',
      skl_precision___multi_str_labeled, sep='')

assert numpy.allclose(
    pandas_precision___multi_str_labeled,
    adf_precision___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_precision___multi_str_labeled,
    spark_precision___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_precision___multi_str_labeled],
    skl_precision___multi_str_labeled,
    equal_nan=False)


print('\n\nRECALL:')


recall___bin_int_labeled = \
    Recall(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_recall___bin_int_labeled = \
    recall___bin_int_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_recall___bin_int_labeled = \
    recall___bin_int_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_recall___bin_int_labeled = \
    recall___bin_int_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_recall___bin_int_labeled = \
    skl_metrics.recall_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n1. Recall (bin, int, labeled):\n\n',
      pandas_recall___bin_int_labeled, '\n\n',
      adf_recall___bin_int_labeled, '\n\n',
      spark_recall___bin_int_labeled, '\n\n',
      skl_recall___bin_int_labeled, sep='')

assert numpy.allclose(
    pandas_recall___bin_int_labeled,
    adf_recall___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_recall___bin_int_labeled,
    spark_recall___bin_int_labeled,
    equal_nan=False) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_recall___bin_int_labeled],
    skl_recall___bin_int_labeled,
    equal_nan=False)


recall___bin_int_unlabeled = \
    Recall(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=2,
        labels=None)

pandas_recall___bin_int_unlabeled = \
    recall___bin_int_unlabeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_recall___bin_int_unlabeled = \
    recall___bin_int_unlabeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_recall___bin_int_unlabeled = \
    recall___bin_int_unlabeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_recall___bin_int_unlabeled = \
    skl_metrics.recall_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n2. Recall (bin, int, unlabeled):\n\n',
      pandas_recall___bin_int_unlabeled, '\n\n',
      adf_recall___bin_int_unlabeled, '\n\n',
      spark_recall___bin_int_unlabeled, '\n\n',
      skl_recall___bin_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_recall___bin_int_unlabeled,
    adf_recall___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_recall___bin_int_unlabeled,
    spark_recall___bin_int_unlabeled,
    equal_nan=False) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_recall___bin_int_unlabeled],
    skl_recall___bin_int_unlabeled,
    equal_nan=False)


recall___bin_str_labeled = \
    Recall(
        label_col=Y_BIN_STR_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_recall___bin_str_labeled = \
    recall___bin_str_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_recall___bin_str_labeled = \
    recall___bin_str_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_recall___bin_str_labeled = \
    recall___bin_str_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_recall___bin_str_labeled = \
    skl_metrics.recall_score(
        y_true=_skl_y_true_bin_str,
        y_pred=_skl_y_pred_bin_str,
        labels=BIN_LABELS,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n3. Recall (bin, int, labeled):\n\n',
      pandas_recall___bin_str_labeled, '\n\n',
      adf_recall___bin_str_labeled, '\n\n',
      spark_recall___bin_str_labeled, '\n\n',
      skl_recall___bin_str_labeled, sep='')

assert numpy.allclose(
    pandas_recall___bin_str_labeled,
    adf_recall___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    pandas_recall___bin_str_labeled,
    spark_recall___bin_str_labeled,
    equal_nan=False) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_recall___bin_str_labeled],
    skl_recall___bin_str_labeled,
    equal_nan=False)


recall___multi_int_labeled = \
    Recall(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_recall___multi_int_labeled = \
    recall___multi_int_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_recall___multi_int_labeled = \
    recall___multi_int_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_recall___multi_int_labeled = \
    recall___multi_int_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_recall___multi_int_labeled = \
    skl_metrics.recall_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n4. Recall (multi, int, labeled):\n\n',
      pandas_recall___multi_int_labeled, '\n\n',
      adf_recall___multi_int_labeled, '\n\n',
      spark_recall___multi_int_labeled, '\n\n',
      skl_recall___multi_int_labeled, sep='')

assert numpy.allclose(
    pandas_recall___multi_int_labeled,
    adf_recall___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_recall___multi_int_labeled,
    spark_recall___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_recall___multi_int_labeled],
    skl_recall___multi_int_labeled,
    equal_nan=False)


recall___multi_int_unlabeled = \
    Recall(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=N_MULTI_CLASSES,
        labels=None)

pandas_recall___multi_int_unlabeled = \
    recall___multi_int_unlabeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_recall___multi_int_unlabeled = \
    recall___multi_int_unlabeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_recall___multi_int_unlabeled = \
    recall___multi_int_unlabeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_recall___multi_int_unlabeled = \
    skl_metrics.recall_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n5. Recall (multi, int, unlabeled):\n\n',
      pandas_recall___multi_int_unlabeled, '\n\n',
      adf_recall___multi_int_unlabeled, '\n\n',
      spark_recall___multi_int_unlabeled, '\n\n',
      skl_recall___multi_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_recall___multi_int_unlabeled,
    adf_recall___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_recall___multi_int_unlabeled,
    spark_recall___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_recall___multi_int_unlabeled],
    skl_recall___multi_int_unlabeled,
    equal_nan=False)


recall___multi_str_labeled = \
    Recall(
        label_col=Y_MULTI_STR_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_recall___multi_str_labeled = \
    recall___multi_str_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_recall___multi_str_labeled = \
    recall___multi_str_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_recall___multi_str_labeled = \
    recall___multi_str_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_recall___multi_str_labeled = \
    skl_metrics.recall_score(
        y_true=_skl_y_true_multi_str,
        y_pred=_skl_y_pred_multi_str,
        labels=MULTI_LABELS,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n6. Recall (multi, str, labeled):\n\n',
      pandas_recall___multi_str_labeled, '\n\n',
      adf_recall___multi_str_labeled, '\n\n',
      spark_recall___multi_str_labeled, '\n\n',
      skl_recall___multi_str_labeled, sep='')

assert numpy.allclose(
    pandas_recall___multi_str_labeled,
    adf_recall___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_recall___multi_str_labeled,
    spark_recall___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_recall___multi_str_labeled],
    skl_recall___multi_str_labeled,
    equal_nan=False)


print('\n\nF1:')


f1___bin_int_labeled = \
    F1(label_col=Y_BIN_INT_COL,
       score_col=SCORE_BIN_COL,
       n_classes=None,
       labels=BIN_LABELS)

pandas_f1___bin_int_labeled = \
    f1___bin_int_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_f1___bin_int_labeled = \
    f1___bin_int_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_f1___bin_int_labeled = \
    f1___bin_int_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_f1___bin_int_labeled = \
    skl_metrics.f1_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n1. F1 (bin, int, labeled):\n\n',
      pandas_f1___bin_int_labeled, '\n\n',
      adf_f1___bin_int_labeled, '\n\n',
      spark_f1___bin_int_labeled, '\n\n',
      skl_f1___bin_int_labeled, sep='')

assert numpy.allclose(
    pandas_f1___bin_int_labeled,
    adf_f1___bin_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_f1___bin_int_labeled,
    spark_f1___bin_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_f1___bin_int_labeled],
    skl_f1___bin_int_labeled,
    equal_nan=False)


f1___bin_int_unlabeled = \
    F1(label_col=Y_BIN_INT_COL,
       score_col=SCORE_BIN_COL,
       n_classes=2,
       labels=None)

pandas_f1___bin_int_unlabeled = \
    f1___bin_int_unlabeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_f1___bin_int_unlabeled = \
    f1___bin_int_unlabeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_f1___bin_int_unlabeled = \
    f1___bin_int_unlabeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_f1___bin_int_unlabeled = \
    skl_metrics.f1_score(
        y_true=_skl_y_true_bin_int,
        y_pred=_skl_y_pred_bin_int,
        labels=None,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n2. F1 (bin, int, unlabeled):\n\n',
      pandas_f1___bin_int_unlabeled, '\n\n',
      adf_f1___bin_int_unlabeled, '\n\n',
      spark_f1___bin_int_unlabeled, '\n\n',
      skl_f1___bin_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_f1___bin_int_unlabeled,
    adf_f1___bin_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_f1___bin_int_unlabeled,
    spark_f1___bin_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_f1___bin_int_unlabeled],
    skl_f1___bin_int_unlabeled,
    equal_nan=False)


f1___bin_str_labeled = \
    F1(label_col=Y_BIN_STR_COL,
       score_col=SCORE_BIN_COL,
       n_classes=None,
       labels=BIN_LABELS)

pandas_f1___bin_str_labeled = \
    f1___bin_str_labeled(
        _uncache_eval_metrics(df),
        BIN_THRESHOLD)

adf_f1___bin_str_labeled = \
    f1___bin_str_labeled(
        _uncache_eval_metrics(adf),
        BIN_THRESHOLD)

spark_f1___bin_str_labeled = \
    f1___bin_str_labeled(
        _uncache_eval_metrics(sdf),
        BIN_THRESHOLD)

skl_f1___bin_str_labeled = \
    skl_metrics.f1_score(
        y_true=_skl_y_true_bin_str,
        y_pred=_skl_y_pred_bin_str,
        labels=BIN_LABELS,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n3. F1 (bin, int, labeled):\n\n',
      pandas_f1___bin_str_labeled, '\n\n',
      adf_f1___bin_str_labeled, '\n\n',
      spark_f1___bin_str_labeled, '\n\n',
      skl_f1___bin_str_labeled, sep='')

assert numpy.allclose(
    pandas_f1___bin_str_labeled,
    adf_f1___bin_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_f1___bin_str_labeled,
    spark_f1___bin_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_f1___bin_str_labeled],
    skl_f1___bin_str_labeled,
    equal_nan=False)


f1___multi_int_labeled = \
    F1(label_col=Y_MULTI_INT_COL,
       score_col=SCORE_MULTI_COL,
       n_classes=None,
       labels=MULTI_LABELS)

pandas_f1___multi_int_labeled = \
    f1___multi_int_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_f1___multi_int_labeled = \
    f1___multi_int_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_f1___multi_int_labeled = \
    f1___multi_int_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_f1___multi_int_labeled = \
    skl_metrics.f1_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n4. F1 (multi, int, labeled):\n\n',
      pandas_f1___multi_int_labeled, '\n\n',
      adf_f1___multi_int_labeled, '\n\n',
      spark_f1___multi_int_labeled, '\n\n',
      skl_f1___multi_int_labeled, sep='')

assert numpy.allclose(
    pandas_f1___multi_int_labeled,
    adf_f1___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_f1___multi_int_labeled,
    spark_f1___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_f1___multi_int_labeled],
    skl_f1___multi_int_labeled,
    equal_nan=False)


f1___multi_int_unlabeled = \
    F1(label_col=Y_MULTI_INT_COL,
       score_col=SCORE_MULTI_COL,
       n_classes=N_MULTI_CLASSES,
       labels=None)

pandas_f1___multi_int_unlabeled = \
    f1___multi_int_unlabeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_f1___multi_int_unlabeled = \
    f1___multi_int_unlabeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_f1___multi_int_unlabeled = \
    f1___multi_int_unlabeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_f1___multi_int_unlabeled = \
    skl_metrics.f1_score(
        y_true=_skl_y_true_multi_int,
        y_pred=_skl_y_pred_multi_int,
        labels=range(N_MULTI_CLASSES),
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n5. F1 (multi, int, unlabeled):\n\n',
      pandas_f1___multi_int_unlabeled, '\n\n',
      adf_f1___multi_int_unlabeled, '\n\n',
      spark_f1___multi_int_unlabeled, '\n\n',
      skl_f1___multi_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_f1___multi_int_unlabeled,
    adf_f1___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_f1___multi_int_unlabeled,
    spark_f1___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_f1___multi_int_unlabeled],
    skl_f1___multi_int_unlabeled,
    equal_nan=False)


f1___multi_str_labeled = \
    F1(label_col=Y_MULTI_STR_COL,
       score_col=SCORE_MULTI_COL,
       n_classes=None,
       labels=MULTI_LABELS)

pandas_f1___multi_str_labeled = \
    f1___multi_str_labeled(
        _uncache_eval_metrics(df),
        *MULTI_THRESHOLDS)

adf_f1___multi_str_labeled = \
    f1___multi_str_labeled(
        _uncache_eval_metrics(adf),
        *MULTI_THRESHOLDS)

spark_f1___multi_str_labeled = \
    f1___multi_str_labeled(
        _uncache_eval_metrics(sdf),
        *MULTI_THRESHOLDS)

skl_f1___multi_str_labeled = \
    skl_metrics.f1_score(
        y_true=_skl_y_true_multi_str,
        y_pred=_skl_y_pred_multi_str,
        labels=MULTI_LABELS,
        pos_label=None,
        average=None,
        sample_weight=None)

print('\n\n6. F1 (multi, str, labeled):\n\n',
      pandas_f1___multi_str_labeled, '\n\n',
      adf_f1___multi_str_labeled, '\n\n',
      spark_f1___multi_str_labeled, '\n\n',
      skl_f1___multi_str_labeled, sep='')

assert numpy.allclose(
    pandas_f1___multi_str_labeled,
    adf_f1___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_f1___multi_str_labeled,
    spark_f1___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    [(0 if numpy.isnan(v) else v)
     for v in pandas_f1___multi_str_labeled],
    skl_f1___multi_str_labeled,
    equal_nan=False)


# print('\n\nAREA UNDER PRECISION-RECALL CURVE:')


# pr_auc___bin_int_labeled = \
#     PR_AuC(
#         label_col=Y_BIN_INT_COL,
#         score_col=SCORE_BIN_COL,
#         n_classes=None,
#         labels=BIN_LABELS)

# pandas_pr_auc___bin_int_labeled = \
#     pr_auc___bin_int_labeled(
#         _uncache_eval_metrics(df))

# adf_pr_auc___bin_int_labeled = \
#     pr_auc___bin_int_labeled(
#         _uncache_eval_metrics(adf))

# spark_pr_auc___bin_int_labeled = \
#     pr_auc___bin_int_labeled(
#         _uncache_eval_metrics(sdf))

# print('\n\n1. PR AuC (bin, int, labeled):\n\n',
#       pandas_pr_auc___bin_int_labeled, '\n\n',
#       adf_pr_auc___bin_int_labeled, '\n\n',
#       spark_pr_auc___bin_int_labeled, sep='')

# assert numpy.allclose(
#     pandas_pr_auc___bin_int_labeled,
#     adf_pr_auc___bin_int_labeled,
#     equal_nan=True) \
#   and numpy.allclose(
#     pandas_pr_auc___bin_int_labeled,
#     spark_pr_auc___bin_int_labeled,
#     equal_nan=True)


# pr_auc___bin_int_unlabeled = \
#     PR_AuC(
#         label_col=Y_BIN_INT_COL,
#         score_col=SCORE_BIN_COL,
#         n_classes=2,
#         labels=None)

# pandas_pr_auc___bin_int_unlabeled = \
#     pr_auc___bin_int_unlabeled(
#         _uncache_eval_metrics(df))

# adf_pr_auc___bin_int_unlabeled = \
#     pr_auc___bin_int_unlabeled(
#         _uncache_eval_metrics(adf))

# spark_pr_auc___bin_int_unlabeled = \
#     pr_auc___bin_int_unlabeled(
#         _uncache_eval_metrics(sdf))

# print('\n\n2. PR AuC (bin, int, unlabeled):\n\n',
#       pandas_pr_auc___bin_int_unlabeled, '\n\n',
#       adf_pr_auc___bin_int_unlabeled, '\n\n',
#       spark_pr_auc___bin_int_unlabeled, sep='')

# assert numpy.allclose(
#     pandas_pr_auc___bin_int_unlabeled,
#     adf_pr_auc___bin_int_unlabeled,
#     equal_nan=True) \
#    and numpy.allclose(
#     pandas_pr_auc___bin_int_unlabeled,
#     spark_pr_auc___bin_int_unlabeled,
#     equal_nan=True)


# pr_auc___bin_str_labeled = \
#     PR_AuC(
#         label_col=Y_BIN_STR_COL,
#         score_col=SCORE_BIN_COL,
#         n_classes=None,
#         labels=BIN_LABELS)

# pandas_pr_auc___bin_str_labeled = \
#     pr_auc___bin_str_labeled(
#         _uncache_eval_metrics(df))

# adf_pr_auc___bin_str_labeled = \
#     pr_auc___bin_str_labeled(
#         _uncache_eval_metrics(adf))

# spark_pr_auc___bin_str_labeled = \
#     pr_auc___bin_str_labeled(
#         _uncache_eval_metrics(sdf))

# print('\n\n3. PR AuC (bin, str, labeled):\n\n',
#       pandas_pr_auc___bin_str_labeled, '\n\n',
#       adf_pr_auc___bin_str_labeled, '\n\n',
#       spark_pr_auc___bin_str_labeled, sep='')

# assert numpy.allclose(
#     pandas_pr_auc___bin_str_labeled,
#     adf_pr_auc___bin_str_labeled,
#     equal_nan=True) \
#    and numpy.allclose(
#     pandas_pr_auc___bin_str_labeled,
#     spark_pr_auc___bin_str_labeled,
#     equal_nan=True)


# pr_auc___multi_int_labeled = \
#     PR_AuC(
#         label_col=Y_MULTI_INT_COL,
#         score_col=SCORE_MULTI_COL,
#         n_classes=None,
#         labels=MULTI_LABELS)

# pandas_pr_auc___multi_int_labeled = \
#     pr_auc___multi_int_labeled(
#         _uncache_eval_metrics(df))

# adf_pr_auc___multi_int_labeled = \
#     pr_auc___multi_int_labeled(
#         _uncache_eval_metrics(adf))

# spark_pr_auc___multi_int_labeled = \
#     pr_auc___multi_int_labeled(
#         _uncache_eval_metrics(sdf))

# print('\n\n4. PR AuC (multi, int, labeled):\n\n',
#       pandas_pr_auc___multi_int_labeled, '\n\n',
#       adf_pr_auc___multi_int_labeled, '\n\n',
#       spark_pr_auc___multi_int_labeled, sep='')

# assert numpy.allclose(
#     pandas_pr_auc___multi_int_labeled,
#     adf_pr_auc___multi_int_labeled,
#     equal_nan=True) \
#    and numpy.allclose(
#     pandas_pr_auc___multi_int_labeled,
#     spark_pr_auc___multi_int_labeled,
#     equal_nan=True)


# pr_auc___multi_int_unlabeled = \
#     PR_AuC(
#         label_col=Y_MULTI_INT_COL,
#         score_col=SCORE_MULTI_COL,
#         n_classes=N_MULTI_CLASSES,
#         labels=None)

# pandas_pr_auc___multi_int_unlabeled = \
#     pr_auc___multi_int_unlabeled(
#         _uncache_eval_metrics(df))

# adf_pr_auc___multi_int_unlabeled = \
#     pr_auc___multi_int_unlabeled(
#         _uncache_eval_metrics(adf))

# spark_pr_auc___multi_int_unlabeled = \
#     pr_auc___multi_int_unlabeled(
#         _uncache_eval_metrics(sdf))

# print('\n\n5. PR AuC (multi, int, unlabeled):\n\n',
#       pandas_pr_auc___multi_int_unlabeled, '\n\n',
#       adf_pr_auc___multi_int_unlabeled, '\n\n',
#       spark_pr_auc___multi_int_unlabeled, sep='')

# assert numpy.allclose(
#     pandas_pr_auc___multi_int_unlabeled,
#     adf_pr_auc___multi_int_unlabeled,
#     equal_nan=True) \
#   and numpy.allclose(
#     pandas_pr_auc___multi_int_unlabeled,
#     spark_pr_auc___multi_int_unlabeled,
#     equal_nan=True)


# pr_auc___multi_str_labeled = \
#     PR_AuC(
#         label_col=Y_MULTI_STR_COL,
#         score_col=SCORE_MULTI_COL,
#         n_classes=None,
#         labels=MULTI_LABELS)

# pandas_pr_auc___multi_str_labeled = \
#     pr_auc___multi_str_labeled(
#         _uncache_eval_metrics(df))

# adf_pr_auc___multi_str_labeled = \
#     pr_auc___multi_str_labeled(
#         _uncache_eval_metrics(adf))

# spark_pr_auc___multi_str_labeled = \
#     pr_auc___multi_str_labeled(
#         _uncache_eval_metrics(sdf))

# print('\n6. PR AuC (multi, str, labeled):\n\n',
#       pandas_pr_auc___multi_str_labeled, '\n\n',
#       adf_pr_auc___multi_str_labeled, '\n\n',
#       spark_pr_auc___multi_str_labeled, sep='')

# assert numpy.allclose(
#     pandas_pr_auc___multi_str_labeled,
#     adf_pr_auc___multi_str_labeled,
#     equal_nan=True) \
#    and numpy.allclose(
#     pandas_pr_auc___multi_str_labeled,
#     spark_pr_auc___multi_str_labeled,
#     equal_nan=True)


print('\n\nAREA UNDER ROC CURVE:')


roc_auc___bin_int_labeled = \
    ROC_AuC(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_roc_auc___bin_int_labeled = \
    roc_auc___bin_int_labeled(
        _uncache_eval_metrics(df))

adf_roc_auc___bin_int_labeled = \
    roc_auc___bin_int_labeled(
        _uncache_eval_metrics(adf))

spark_roc_auc___bin_int_labeled = \
    roc_auc___bin_int_labeled(
        _uncache_eval_metrics(sdf))

print('\n1. ROC AuC (bin, int, labeled):',
      pandas_roc_auc___bin_int_labeled,
      adf_roc_auc___bin_int_labeled,
      spark_roc_auc___bin_int_labeled)

assert numpy.allclose(
    pandas_roc_auc___bin_int_labeled,
    adf_roc_auc___bin_int_labeled,
    equal_nan=True) \
  and numpy.allclose(
    pandas_roc_auc___bin_int_labeled,
    spark_roc_auc___bin_int_labeled,
    equal_nan=True)


roc_auc___bin_int_unlabeled = \
    ROC_AuC(
        label_col=Y_BIN_INT_COL,
        score_col=SCORE_BIN_COL,
        n_classes=2,
        labels=None)

pandas_roc_auc___bin_int_unlabeled = \
    roc_auc___bin_int_unlabeled(
        _uncache_eval_metrics(df))

adf_roc_auc___bin_int_unlabeled = \
    roc_auc___bin_int_unlabeled(
        _uncache_eval_metrics(adf))

spark_roc_auc___bin_int_unlabeled = \
    roc_auc___bin_int_unlabeled(
        _uncache_eval_metrics(sdf))

print('\n2. ROC AuC (bin, int, unlabeled):',
      pandas_roc_auc___bin_int_unlabeled,
      adf_roc_auc___bin_int_unlabeled,
      spark_roc_auc___bin_int_unlabeled)

assert numpy.allclose(
    pandas_roc_auc___bin_int_unlabeled,
    adf_roc_auc___bin_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_roc_auc___bin_int_unlabeled,
    spark_roc_auc___bin_int_unlabeled,
    equal_nan=True)


roc_auc___bin_str_labeled = \
    ROC_AuC(
        label_col=Y_BIN_STR_COL,
        score_col=SCORE_BIN_COL,
        n_classes=None,
        labels=BIN_LABELS)

pandas_roc_auc___bin_str_labeled = \
    roc_auc___bin_str_labeled(
        _uncache_eval_metrics(df))

adf_roc_auc___bin_str_labeled = \
    roc_auc___bin_str_labeled(
        _uncache_eval_metrics(adf))

spark_roc_auc___bin_str_labeled = \
    roc_auc___bin_str_labeled(
        _uncache_eval_metrics(sdf))

print('\n3. ROC AuC (bin, str, labeled):',
      pandas_roc_auc___bin_str_labeled,
      adf_roc_auc___bin_str_labeled,
      spark_roc_auc___bin_str_labeled)

assert numpy.allclose(
    pandas_roc_auc___bin_str_labeled,
    adf_roc_auc___bin_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_roc_auc___bin_str_labeled,
    spark_roc_auc___bin_str_labeled,
    equal_nan=True)


roc_auc___multi_int_labeled = \
    ROC_AuC(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_roc_auc___multi_int_labeled = \
    roc_auc___multi_int_labeled(
        _uncache_eval_metrics(df))

adf_roc_auc___multi_int_labeled = \
    roc_auc___multi_int_labeled(
        _uncache_eval_metrics(adf))

spark_roc_auc___multi_int_labeled = \
    roc_auc___multi_int_labeled(
        _uncache_eval_metrics(sdf))

print('\n\n4. ROC AuC (multi, int, labeled):\n\n',
      pandas_roc_auc___multi_int_labeled, '\n\n',
      adf_roc_auc___multi_int_labeled, '\n\n',
      spark_roc_auc___multi_int_labeled, sep='')

assert numpy.allclose(
    pandas_roc_auc___multi_int_labeled,
    adf_roc_auc___multi_int_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_roc_auc___multi_int_labeled,
    spark_roc_auc___multi_int_labeled,
    equal_nan=True)


roc_auc___multi_int_unlabeled = \
    ROC_AuC(
        label_col=Y_MULTI_INT_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=N_MULTI_CLASSES,
        labels=None)

pandas_roc_auc___multi_int_unlabeled = \
    roc_auc___multi_int_unlabeled(
        _uncache_eval_metrics(df))

adf_roc_auc___multi_int_unlabeled = \
    roc_auc___multi_int_unlabeled(
        _uncache_eval_metrics(adf))

spark_roc_auc___multi_int_unlabeled = \
    roc_auc___multi_int_unlabeled(
        _uncache_eval_metrics(sdf))

print('\n\n5. ROC AuC (multi, int, unlabeled):\n\n',
      pandas_roc_auc___multi_int_unlabeled, '\n\n',
      adf_roc_auc___multi_int_unlabeled, '\n\n',
      spark_roc_auc___multi_int_unlabeled, sep='')

assert numpy.allclose(
    pandas_roc_auc___multi_int_unlabeled,
    adf_roc_auc___multi_int_unlabeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_roc_auc___multi_int_unlabeled,
    spark_roc_auc___multi_int_unlabeled,
    equal_nan=True)


roc_auc___multi_str_labeled = \
    ROC_AuC(
        label_col=Y_MULTI_STR_COL,
        score_col=SCORE_MULTI_COL,
        n_classes=None,
        labels=MULTI_LABELS)

pandas_roc_auc___multi_str_labeled = \
    roc_auc___multi_str_labeled(
        _uncache_eval_metrics(df))

adf_roc_auc___multi_str_labeled = \
    roc_auc___multi_str_labeled(
        _uncache_eval_metrics(adf))

spark_roc_auc___multi_str_labeled = \
    roc_auc___multi_str_labeled(
        _uncache_eval_metrics(sdf))

print('\n\n6. ROC AuC (multi, str, labeled):\n\n',
      pandas_roc_auc___multi_str_labeled, '\n\n',
      adf_roc_auc___multi_str_labeled, '\n\n',
      spark_roc_auc___multi_str_labeled, sep='')

assert numpy.allclose(
    pandas_roc_auc___multi_str_labeled,
    adf_roc_auc___multi_str_labeled,
    equal_nan=True) \
   and numpy.allclose(
    pandas_roc_auc___multi_str_labeled,
    spark_roc_auc___multi_str_labeled,
    equal_nan=True)
