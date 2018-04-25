import abc


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
