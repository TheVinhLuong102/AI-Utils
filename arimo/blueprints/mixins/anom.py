from __future__ import division

import copy
import itertools
import numpy
import pandas

from pyspark.sql import functions
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from arimo.df.from_files import ArrowADF
from arimo.df.spark import SparkADF
from arimo.util.date_time import DATE_COL, MONTH_COL
from arimo.util.iterables import to_iterable


class PPPAnalysesMixIn(object):
    _SGN_PREFIX = 'sgn__'
    _ABS_PREFIX = 'abs__'
    _NEG_PREFIX = 'neg__'
    _POS_PREFIX = 'pos__'
    _SGN_PREFIXES = _SGN_PREFIX, _ABS_PREFIX, _NEG_PREFIX, _POS_PREFIX

    _BENCHMARK_METRICS_ADF_ALIAS = '__BenchmarkMetrics__'

    _INDIV_PREFIX = 'indiv__'
    _GLOBAL_PREFIX = 'global__'
    _INDIV_OR_GLOBAL_PREFIXES = _INDIV_PREFIX, _GLOBAL_PREFIX

    _RAW_METRICS = 'MedAE', 'MAE'   #, 'RMSE'

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
        [(_row_summ_prefix + _ABS_PREFIX + _indiv_or_global_prefix + _ERR_MULT_COLS[_metric])
         for _metric, _indiv_or_global_prefix, _row_summ_prefix in
            itertools.product(_RAW_METRICS, _INDIV_OR_GLOBAL_PREFIXES, _ROW_SUMM_PREFIXES)]

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

    def err_mults(self, df):
        label_var_names = []
        score_col_names = {}

        for label_var_name, component_blueprint_params in self.params.model.component_blueprints.items():
            if (label_var_name in df.columns) and component_blueprint_params.model.ver:
                label_var_names.append(label_var_name)

                score_col_names[label_var_name] = \
                    component_blueprint_params.model.score.raw_score_col_prefix + label_var_name

        benchmark_metric_col_names = {}
        benchmark_metric_col_names_list = []

        for _indiv_or_global_prefix in self._INDIV_OR_GLOBAL_PREFIXES:
            benchmark_metric_col_names[_indiv_or_global_prefix] = {}

            for _raw_metric in (('n',) + self._RAW_METRICS):
                benchmark_metric_col_names[_indiv_or_global_prefix][_raw_metric] = {}

                for label_var_name in label_var_names:
                    benchmark_metric_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name] = \
                        benchmark_metric_col_name = \
                        _indiv_or_global_prefix + _raw_metric + '__' + label_var_name

                    benchmark_metric_col_names_list.append(benchmark_metric_col_name)

        err_mult_col_names = {}
        abs_err_mult_col_names = {}

        for _indiv_or_global_prefix in self._INDIV_OR_GLOBAL_PREFIXES:
            err_mult_col_names[_indiv_or_global_prefix] = {}
            abs_err_mult_col_names[_indiv_or_global_prefix] = {}

            for _raw_metric in self._RAW_METRICS:
                err_mult_col_names[_indiv_or_global_prefix][_raw_metric] = {}
                abs_err_mult_col_names[_indiv_or_global_prefix][_raw_metric] = {}

                for label_var_name in label_var_names:
                    err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name] = {}

                    for _sgn_prefix in self._SGN_PREFIXES:
                        err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][_sgn_prefix] = \
                            err_mult_col = \
                            _sgn_prefix + _indiv_or_global_prefix + self._ERR_MULT_PREFIXES[_raw_metric] + label_var_name

                        if _sgn_prefix == self._ABS_PREFIX:
                            abs_err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name] = err_mult_col

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
                    benchmark_metrics_df[benchmark_metric_col_names[self._INDIV_PREFIX][_raw_metric][label_var_name]] = \
                        benchmark_metrics_df[id_col].map(
                            lambda _id:
                                self.params.benchmark_metrics[label_var_name][self._BY_ID_EVAL_KEY]
                                    .get(_id, {})
                                    .get(_raw_metric))

                    benchmark_metrics_df[benchmark_metric_col_names[self._GLOBAL_PREFIX][_raw_metric][label_var_name]] = \
                        self.params.benchmark_metrics[label_var_name][self._GLOBAL_EVAL_KEY][_raw_metric]

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

                _sgn_err_col_expr = df[label_var_name] - df[score_col_name]

                for _indiv_or_global_prefix in self._INDIV_OR_GLOBAL_PREFIXES:
                    for _raw_metric in self._RAW_METRICS:
                        _sgn_err_mult_col_expr = \
                            _sgn_err_col_expr / \
                            df[benchmark_metric_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name]]
        
                        col_exprs += \
                            [_sgn_err_mult_col_expr
                                .alias(err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._SGN_PREFIX]),
        
                             functions.abs(_sgn_err_mult_col_expr)
                                .alias(err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._ABS_PREFIX]),
        
                             functions.when(df[label_var_name] < df[score_col_name], _sgn_err_mult_col_expr).otherwise(None)
                                .alias(err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._NEG_PREFIX]),
        
                             functions.when(df[label_var_name] > df[score_col_name], _sgn_err_mult_col_expr).otherwise(None)
                                .alias(err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._POS_PREFIX])]

            df = df.select('*', *col_exprs)

        else:
            _is_adf = False

            for label_var_name in label_var_names:
                score_col_name = score_col_names[label_var_name]

                _sgn_err_series = df[label_var_name] - df[score_col_name]
                _neg_chk_series = _sgn_err_series < 0
                _pos_chk_series = _sgn_err_series > 0

                for _raw_metric in self._RAW_METRICS:
                    df[benchmark_metric_col_names[self._INDIV_PREFIX][_raw_metric][label_var_name]] = \
                        df[id_col].map(
                            lambda id:
                                self.params.benchmark_metrics[label_var_name][self._BY_ID_EVAL_KEY]
                                    .get(id, {})
                                    .get(_raw_metric, numpy.nan))

                    df[benchmark_metric_col_names[self._GLOBAL_PREFIX][_raw_metric][label_var_name]] = \
                        self.params.benchmark_metrics[label_var_name][self._GLOBAL_EVAL_KEY][_raw_metric]

                    for _indiv_or_global_prefix in self._INDIV_OR_GLOBAL_PREFIXES:
                        df[err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._SGN_PREFIX]] = \
                            _sgn_err_mult_series = \
                            _sgn_err_series / \
                            df[benchmark_metric_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name]]

                        df[err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._ABS_PREFIX]] = \
                            _sgn_err_mult_series.abs()

                        df.loc[_neg_chk_series,
                               err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._NEG_PREFIX]] = \
                            _sgn_err_mult_series.loc[_neg_chk_series]

                        df.loc[_pos_chk_series,
                               err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name][self._POS_PREFIX]] = \
                            _sgn_err_mult_series.loc[_pos_chk_series]

        n_label_vars = len(label_var_names)

        if _is_adf:
            _row_summ_col_exprs = []

            for _raw_metric in self._RAW_METRICS:
                for _indiv_or_global_prefix in self._INDIV_OR_GLOBAL_PREFIXES:
                    _abs_err_mult_col_names = \
                        list(abs_err_mult_col_names[_indiv_or_global_prefix][_raw_metric].values())

                    _row_summ_col_name_body = \
                        self._ABS_PREFIX + _indiv_or_global_prefix + self._ERR_MULT_COLS[_raw_metric]

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
                             functions.log(df[_abs_err_mult_col_name]).alias(_rowSumOfLog_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowHigh_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowLow_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowMean_summ_col_name),
                             df[_abs_err_mult_col_name].alias(_rowGMean_summ_col_name)]

            return df.select('*', *_row_summ_col_exprs)

        else:
            if isinstance(df, ArrowADF):
                df = df.toPandas()
            
            for _raw_metric in self._RAW_METRICS:
                for _indiv_or_global_prefix in self._INDIV_OR_GLOBAL_PREFIXES:
                    _row_summ_col_name_body = \
                        self._ABS_PREFIX + _indiv_or_global_prefix + self._ERR_MULT_COLS[_raw_metric]

                    _rowEuclNorm_summ_col_name = self._rowEuclNorm_PREFIX + _row_summ_col_name_body
                    _rowSumOfLog_summ_col_name = self._rowSumOfLog_PREFIX + _row_summ_col_name_body
                    _rowHigh_summ_col_name = self._rowHigh_PREFIX + _row_summ_col_name_body
                    _rowLow_summ_col_name = self._rowLow_PREFIX + _row_summ_col_name_body
                    _rowMean_summ_col_name = self._rowMean_PREFIX + _row_summ_col_name_body
                    _rowGMean_summ_col_name = self._rowGMean_PREFIX + _row_summ_col_name_body

                    if n_label_vars > 1:
                        abs_err_mults_df = \
                            df[list(abs_err_mult_col_names[_indiv_or_global_prefix][_raw_metric].values())]

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
                            abs_err_mult_col_names[_indiv_or_global_prefix][_raw_metric][label_var_name]

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
                    [(_sgn + _indiv_or_global_prefix + cls._ERR_MULT_PREFIXES[_metric] + label_var_name)
                     for _metric, _indiv_or_global_prefix, _sgn in
                        itertools.product(cls._RAW_METRICS, cls._INDIV_OR_GLOBAL_PREFIXES, cls._SGN_PREFIXES)]

        if isinstance(df_w_err_mults, SparkADF):
            from arimo.blueprints.base import _SupervisedBlueprintABC

            col_strs = []

            for col_name in cols_to_agg:
                assert col_name in df_w_err_mults.columns

                col_strs += \
                    ['PERCENTILE_APPROX(GREATEST(LEAST({0}, {1}), -{1}), 0.5) AS {2}{0}'.format(col_name, clip, cls._dailyMed_PREFIX),
                     'AVG(GREATEST(LEAST({0}, {1}), -{1})) AS {2}{0}'.format(col_name, clip, cls._dailyMean_PREFIX),
                     'MAX(GREATEST(LEAST({0}, {1}), -{1})) AS {2}{0}'.format(col_name, clip, cls._dailyMax_PREFIX),
                     'MIN(GREATEST(LEAST({0}, {1}), -{1})) AS {2}{0}'.format(col_name, clip, cls._dailyMin_PREFIX)]

            for _indiv_or_global_prefix in cls._INDIV_OR_GLOBAL_PREFIXES:
                for _raw_metric in cls._RAW_METRICS:
                    for label_var_name in label_var_names:
                        if label_var_name in df_w_err_mults.columns:
                            _metric_col_name = _indiv_or_global_prefix + _raw_metric + '__' + label_var_name
                            cols_to_agg.append(_metric_col_name)
                            col_strs.append('AVG({0}) AS {0}'.format(_metric_col_name))

            return df_w_err_mults(
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

        else:
            if df_w_err_mults[time_col].dtype != 'datetime64[ns]':
                df_w_err_mults[time_col] = pandas.DatetimeIndex(df_w_err_mults[time_col])

            def f(group_df):
                cols = [id_col, DATE_COL]

                _first_row = group_df.iloc[0]

                d = {id_col: _first_row[id_col],
                     DATE_COL: _first_row[time_col]}

                for _indiv_or_global_prefix in cls._INDIV_OR_GLOBAL_PREFIXES:
                    for _raw_metric in cls._RAW_METRICS:
                        for label_var_name in label_var_names:
                            if label_var_name in df_w_err_mults.columns:
                                _metric_col_name = _indiv_or_global_prefix + _raw_metric + '__' + label_var_name

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

            return df_w_err_mults.groupby(
                    by=[df_w_err_mults[id_col], df_w_err_mults[time_col].dt.date],
                    axis='index',
                    level=None,
                    as_index=False,
                    sort=True,
                    group_keys=True,
                    squeeze=False).apply(f)

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

        return daily_err_mults_df.drop(
                columns=daily_err_mult_summ_col_names,
                level=None,
                inplace=False,
                errors='raise')
