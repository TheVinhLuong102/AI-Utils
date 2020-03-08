__all__ = (
    'DATE_COL', 'MONTH_COL',

    '_T_ORD_COL', '_T_DELTA_COL', '_T_REL_AUX_COLS',
    '_T_HoY_COL', '_T_QoY_COL', '_T_MoY_COL', '_T_PoY_COL',   # '_T_WoY_COL', '_T_DoY_COL'
    '_T_QoH_COL', '_T_MoH_COL', '_T_PoH_COL',
    '_T_MoQ_COL', '_T_PoQ_COL',
    '_T_WoM_COL', '_T_DoM_COL', '_T_PoM_COL',
    '_T_DoW_COL', '_T_PoW_COL',
    '_T_HoD_COL', '_T_PoD_COL',
    '_T_COMPONENT_AUX_COLS', '_T_AUX_COLS', '_T_CAT_AUX_COLS', '_T_NUM_AUX_COLS',
    '_PRED_VARS_INCL_T_CAT_AUX_COLS', '_PRED_VARS_INCL_T_NUM_AUX_COLS', '_PRED_VARS_INCL_T_AUX_COLS',

    'gen_aux_cols',

    'month_end', 'month_str',

    'now'
)


import datetime
from dateutil.relativedelta import relativedelta
import pandas


DATE_COL = 'date'
MONTH_COL = 'month'


_T_ORD_COL = '__tOrd__'
_T_DELTA_COL = '__tDelta__'

_T_REL_AUX_COLS = _T_ORD_COL, _T_DELTA_COL


_T_HoY_COL = '__tHoY__'   # Half of Year
_T_QoY_COL = '__tQoY__'   # Quarter of Year
_T_MoY_COL = '__tMoY__'   # Month of Year
# _T_WoY_COL = '__tWoY__'   # Week of Year
# _T_DoY_COL = '__tDoY__'   # Day of Year
_T_PoY_COL = '__tPoY__'   # Part/Proportion/Fraction of Year

_T_QoH_COL = '__tQoH__'   # Quarter of Half-Year
_T_MoH_COL = '__tMoH__'   # Month of Half-Year
_T_PoH_COL = '__tPoH__'   # Part/Proportion/Fraction of Half-Year

_T_MoQ_COL = '__tMoQ__'   # Month of Quarter
_T_PoQ_COL = '__tPoQ__'   # Part/Proportion/Fraction of Quarter

_T_WoM_COL = '__tWoM__'   # Week of Month
_T_DoM_COL = '__tDoM__'   # Day of Month
_T_PoM_COL = '__tPoM__'   # Part/Proportion/Fraction of Month

_T_DoW_COL = '__tDoW__'   # Day of Week
_T_PoW_COL = '__tPoW__'   # Part/Proportion/Fraction of Week

_T_HoD_COL = '__tHoD__'   # Hour of Day
_T_PoD_COL = '__tPoD__'   # Part/Proportion/Fraction of Day

_T_COMPONENT_AUX_COLS = \
    (_T_HoY_COL, _T_QoY_COL, _T_MoY_COL, _T_PoY_COL,   # _T_WoY_COL, _T_DoY_COL,
     _T_QoH_COL, _T_MoH_COL, _T_PoH_COL,
     _T_MoQ_COL, _T_PoQ_COL,
     _T_WoM_COL, _T_DoM_COL, _T_PoM_COL,
     _T_DoW_COL, _T_PoW_COL,
     _T_HoD_COL, _T_PoD_COL)


_T_AUX_COLS = _T_REL_AUX_COLS + _T_COMPONENT_AUX_COLS


_T_CAT_AUX_COLS = \
    _T_HoY_COL, _T_QoY_COL, _T_MoY_COL, \
    _T_QoH_COL, _T_MoH_COL, \
    _T_MoQ_COL, \
    _T_WoM_COL, \
    _T_DoW_COL, \
    _T_HoD_COL

_T_NUM_AUX_COLS = \
    (_T_DELTA_COL,
     _T_PoY_COL,   # _T_WoY_COL, _T_DoY_COL,
     _T_PoH_COL,
     _T_PoQ_COL,
     _T_DoM_COL, _T_PoM_COL,
     _T_PoW_COL,
     _T_PoD_COL)


_PRED_VARS_INCL_T_CAT_AUX_COLS = \
    (_T_HoY_COL, _T_QoY_COL, _T_MoY_COL,
     _T_QoH_COL, _T_MoH_COL,
     _T_MoQ_COL,
     _T_WoM_COL,
     _T_DoW_COL)

_PRED_VARS_INCL_T_NUM_AUX_COLS = \
    (_T_DELTA_COL,
     _T_PoY_COL,
     _T_PoH_COL,
     _T_PoQ_COL,
     _T_PoM_COL,
     _T_PoW_COL,
     _T_PoD_COL)

_PRED_VARS_INCL_T_AUX_COLS = _PRED_VARS_INCL_T_CAT_AUX_COLS + _PRED_VARS_INCL_T_NUM_AUX_COLS


def gen_aux_cols(
        df: pandas.DataFrame,   # TODO Py3.8: positional-only
        *, i_col: str = None, t_col: str = 't')\
        -> pandas.DataFrame:
    assert t_col in df.columns, \
        '*** "{}" NOT AMONG {} ***'.format(t_col, df.columns.tolist())

    if df[t_col].dtype != 'datetime64[ns]':
        df[t_col] = pandas.DatetimeIndex(df[t_col])

    if i_col:
        assert i_col in df.columns

        df.sort_values(
            by=[i_col, t_col],
            axis='index',
            ascending=True,
            kind='quicksort',
            na_position='last',
            inplace=True)

        g = df.groupby(
            by=i_col,
            axis='index',
            level=None,
            as_index=True,
            sort=False,
            group_keys=True,
            squeeze=False)

        df[_T_ORD_COL] = \
            g.cumcount()

        df[_T_DELTA_COL] = \
            (df[t_col] -
             g[t_col].shift(
                periods=1,
                freq=None,
                axis='index')) \
            .dt.total_seconds()

        df.reset_index(
            level=None,
            drop=True,
            inplace=True,
            col_level=0,
            col_fill='')

    t_attr_access = df[t_col].dt

    if DATE_COL not in df.columns:
        df[DATE_COL] = t_attr_access.date

    if MONTH_COL not in df.columns:
        df[MONTH_COL] = df[DATE_COL].map(lambda d: str(d)[:7])

    if (_T_QoY_COL not in df.columns) or (df[_T_QoY_COL].dtype != int):
        df[_T_QoY_COL] = _ = t_attr_access.quarter
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_HoY_COL not in df.columns) or (df[_T_HoY_COL].dtype != int):
        df[_T_HoY_COL] = _ = (df[_T_QoY_COL] - 1) // 2 + 1
        assert _.dtype == int, '*** {} from {} ***'.format(_, df[_T_QoY_COL])

    if (_T_MoY_COL not in df.columns) or (df[_T_MoY_COL].dtype != int):
        df[_T_MoY_COL] = _ = t_attr_access.month
        assert _.dtype == int, '*** {} ***'.format(_)

    # if (_T_WoY_COL not in df.columns) or (df[_T_WoY_COL].dtype != int):
    #     df[_T_WoY_COL] = _ = t_attr_access.weekofyear
    #     assert _.dtype == int, '*** {} ***'.format(_)

    # if (_T_DoY_COL not in df.columns) and (df[_T_DoY_COL].dtype != int):
    #     df[_T_DoY_COL] = _ = t_attr_access.dayofyear
    #     assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_PoY_COL not in df.columns) or (df[_T_PoY_COL].dtype != float):
        df[_T_PoY_COL] = _ = df[_T_MoY_COL] / 12
        assert _.dtype == float, '*** {} ***'.format(_)

    if (_T_QoH_COL not in df.columns) or (df[_T_QoH_COL].dtype != int):
        df[_T_QoH_COL] = _ = (df[_T_QoY_COL] - 1) % 2 + 1
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_MoH_COL not in df.columns) or (df[_T_MoH_COL].dtype != int):
        df[_T_MoH_COL] = _ = (df[_T_MoY_COL] - 1) % 6 + 1
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_PoH_COL not in df.columns) or (df[_T_PoH_COL].dtype != float):
        df[_T_PoH_COL] = _ = df[_T_MoH_COL] / 6
        assert _.dtype == float, '*** {} ***'.format(_)

    if (_T_MoQ_COL not in df.columns) or (df[_T_MoQ_COL].dtype != int):
        df[_T_MoQ_COL] = _ = (df[_T_MoY_COL] - 1) % 3 + 1
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_PoQ_COL not in df.columns) or (df[_T_PoQ_COL].dtype != float):
        df[_T_PoQ_COL] = _ = df[_T_MoQ_COL] / 3
        assert _.dtype == float, '*** {} ***'.format(_)

    if (_T_DoM_COL not in df.columns) or (df[_T_DoM_COL].dtype != int):
        df[_T_DoM_COL] = _ = t_attr_access.day
        assert _.dtype == int

    if (_T_WoM_COL not in df.columns) or (df[_T_WoM_COL].dtype != int):
        df[_T_WoM_COL] = _ = (df[_T_DoM_COL] - 1) // 7 + 1
        df.loc[df[_T_WoM_COL] > 4, _T_WoM_COL] = 4
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_PoM_COL not in df.columns) or (df[_T_PoM_COL].dtype != float):
        df[_T_PoM_COL] = _ = df[_T_DoM_COL] / t_attr_access.days_in_month
        assert _.dtype == float, '*** {} ***'.format(_)

    if (_T_DoW_COL not in df.columns) or (df[_T_DoW_COL].dtype != int):
        df[_T_DoW_COL] = _ = t_attr_access.dayofweek + 1
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_PoW_COL not in df.columns) or (df[_T_PoW_COL].dtype != float):
        df[_T_PoW_COL] = _ = df[_T_DoW_COL] / 7
        assert _.dtype == float, '*** {} ***'.format(_)
        
    if (_T_HoD_COL not in df.columns) or (df[_T_HoD_COL].dtype != int):
        df[_T_HoD_COL] = _ = t_attr_access.hour
        assert _.dtype == int, '*** {} ***'.format(_)

    if (_T_PoD_COL not in df.columns) or (df[_T_PoD_COL].dtype != float):
        df[_T_PoD_COL] = _ = df[_T_HoD_COL] / 24
        assert _.dtype == float, '*** {} ***'.format(_)

    return df[([i_col,
                t_col,
                _T_ORD_COL,
                _T_DELTA_COL]
               if i_col
               else [t_col]) +
              list(_T_COMPONENT_AUX_COLS) +
              [col for col in df.columns
                   if (col != i_col) and (col != t_col) and (col not in _T_AUX_COLS)]]


def month_end(date_or_month_str: str) -> datetime.date:
    month_start_date = \
        datetime.datetime.strptime(date_or_month_str[:7] + '-01', "%Y-%m-%d")

    return (month_start_date + relativedelta(months=1, days=-1)).date()


def month_str(date_or_month_str, n_months_offset=0) -> str:
    month_start_date = \
        datetime.datetime.strptime(date_or_month_str[:7] + '-01', "%Y-%m-%d")

    return str((month_start_date + relativedelta(months=n_months_offset))
               if n_months_offset
               else month_start_date)[:7]


def now(*,
        strf='%Y-%m-%d %H:%M:%S',
        utc=True) \
        -> (str, datetime.date):
    dt = datetime.datetime.utcnow() \
        if utc \
        else datetime.datetime.now()

    return dt.strftime(strf) \
        if strf \
      else dt
