import pandas

from arimo.df.spark import ADF
from arimo.util.spark_sql_types import _BIGINT_TYPE, _INT_TYPE


ts_adf = ADF.create(
    data=pandas.DataFrame(
        data=dict(
            id=3 * ('i0',) + 3 * ('i1',) + 3 * ('i2',),
            t=3 * ('2018-01-01 06:06:06', '2018-01-02 12:12:12', '2018-01-03 18:18:18'),
            x=range(9))),
    iCol='id', tCol='t')

cols = ts_adf.columns
ts_adf_schema = ts_adf.schema


for t_aux_col in ts_adf.tAuxCols:
    _ts_adf_schema = ts_adf.rm(t_aux_col).schema
    if _ts_adf_schema != ts_adf_schema:
        assert [col for col in cols if _ts_adf_schema[col] != ts_adf_schema[col]] == [t_aux_col]
        assert (ts_adf_schema[t_aux_col].dataType.simpleString() == _INT_TYPE) \
           and (_ts_adf_schema[t_aux_col].dataType.simpleString() == _BIGINT_TYPE)
