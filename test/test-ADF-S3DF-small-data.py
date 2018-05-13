from __future__ import division, print_function

import numpy
import pandas
from pympler.asizeof import asizeof
import sys
import tempfile

from arimo.df.s3 import S3DF
from arimo.df.spark import ADF
from arimo.util.aws import s3
from arimo.util.date_time import _T_DELTA_COL, _T_COMPONENT_AUX_COLS
from arimo.util import fs
import arimo.__debug__

from arimo.IoT.PredMaint import Project

from PanasonicColdChain import _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY
from data import SMALL_DATA_S3_PARQUET_PATH


SMALL_SORTED_DATA_S3_PARQUET_PATH = \
    's3://arimo-panasonic-ap/data/test-resources/DISP_CASE---ex_display_case---SORTED.parquet'


TS_ADF_DATA_TRANSFORM_PATH = '/tmp/test-TSADF-TimeSerDataTransforms'
TS_S3DF_DATA_TRANSFORM_PATH = ''


BYTES_IN_GB = 10 ** 9


project = \
    Project(
        params={
            'db.user': 'arimo_cc',
            'db.password': 'k3qnhMxNVxjTzF2',

            'db.measure.url': 'arimoc-coldchain.c1t5cncmtcff.us-east-1.redshift.amazonaws.com:5439/measurement',

            'db.health.url': 'arimoc-coldchain.c1t5cncmtcff.us-east-1.redshift.amazonaws.com:5439/measurement',

            'db.anom.url': 'arimoc-coldchain.c1t5cncmtcff.us-east-1.redshift.amazonaws.com:5439/measurement',

            'db.admin': dict(
                host='panacc-iotdataadmin.cgp5wm6rcqwy.us-east-1.rds.amazonaws.com',
                user='arimo', password='arimoiscool',
                name='PanaCC_IoTDataAdmin'),

            's3.bucket': 'arimo-panasonic-ap',
            's3.equipment_data_dir_prefix': 'data/CombinedConfigMeasure',

            's3.access_key_id': _AWS_ACCESS_KEY_ID,
            's3.secret_access_key': _AWS_SECRET_ACCESS_KEY})


bp = project._blueprint(
    equipment_general_type_name='disp_case',
    equipment_unique_type_names='ex_display_case',
    monitored_n_excluded_measure_data_fields=
        dict(inside_temperature=['defrosting_temperature',
                                 'discharge_temperature']),
    timeser_input_len=30, incl_time_features=True,
    params={}, verbose=True)


adf = ADF.load(
    path=SMALL_DATA_S3_PARQUET_PATH, format='parquet',
    aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
    verbose=True)


if '__disp_cases__' in adf.columns:
    adf.rm('__disp_cases__', inplace=True)
    adf.save(
        path=SMALL_DATA_S3_PARQUET_PATH, format='parquet', partitionBy='date',
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        switch=True, verbose=True)


ts_adf = \
    adf('*',
        iCol=project._EQUIPMENT_INSTANCE_ID_COL_NAME,
        tCol=project._DATE_TIME_COL_NAME)


cols_excl_tDelta = \
    [project._EQUIPMENT_INSTANCE_ID_COL_NAME, project._DATE_TIME_COL_NAME] + \
    list(_T_COMPONENT_AUX_COLS) + \
    sorted(ts_adf.contentCols)

cols_incl_tDelta = \
    cols_excl_tDelta + \
    [_T_DELTA_COL]


_tmp_path = tempfile.mkdtemp()

ts_adf.sparkDF.write.save(path=_tmp_path, format='parquet', mode='overwrite')

fs.get(
    from_hdfs=_tmp_path, to_local=_tmp_path,
    is_dir=True, overwrite=True, _mv=True,
    must_succeed=True)

s3.sync(
    from_dir_path=_tmp_path,
    to_dir_path=SMALL_SORTED_DATA_S3_PARQUET_PATH,
    access_key_id=_AWS_ACCESS_KEY_ID, secret_access_key=_AWS_SECRET_ACCESS_KEY,
    quiet=True, delete=True, verbose=True)

fs.rm(
    path=_tmp_path,
    hdfs=False,
    is_dir=True)


ts_s3df_0 = \
    S3DF(paths=SMALL_SORTED_DATA_S3_PARQUET_PATH,
         aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY)

sys.getsizeof(ts_s3df_0) / BYTES_IN_GB
asizeof(ts_s3df_0) / BYTES_IN_GB


ts_df_0 = \
    ts_s3df_0.collect()[cols_incl_tDelta] \
        .sort_values(
            by=[project._EQUIPMENT_INSTANCE_ID_COL_NAME, project._DATE_TIME_COL_NAME],
            axis='index',
            ascending=True,
            kind='quicksort',
            na_position='last',
            inplace=False) \
        .reset_index(
            level=None,
            drop=True,
            inplace=False,
            col_level=0,
            col_fill='')

ts_df_0.memory_usage().sum() / BYTES_IN_GB
sys.getsizeof(ts_df_0) / BYTES_IN_GB
asizeof(ts_df_0) / BYTES_IN_GB

# re-measure S3DF cache size
sys.getsizeof(ts_s3df_0) / BYTES_IN_GB
asizeof(ts_s3df_0) / BYTES_IN_GB


ts_s3df = \
    S3DF(paths=SMALL_DATA_S3_PARQUET_PATH,
         aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
         i_col=project._EQUIPMENT_INSTANCE_ID_COL_NAME, t_col=project._DATE_TIME_COL_NAME)

sys.getsizeof(ts_s3df) / BYTES_IN_GB
asizeof(ts_s3df) / BYTES_IN_GB


ts_df = \
    ts_s3df.collect()[cols_incl_tDelta] \
        .sort_values(
            by=[project._EQUIPMENT_INSTANCE_ID_COL_NAME, project._DATE_TIME_COL_NAME],
            axis='index',
            ascending=True,
            kind='quicksort',
            na_position='last',
            inplace=False) \
        .reset_index(
            level=None,
            drop=True,
            inplace=False,
            col_level=0,
            col_fill='')

ts_df.memory_usage().sum() / BYTES_IN_GB
sys.getsizeof(ts_df) / BYTES_IN_GB
asizeof(ts_df) / BYTES_IN_GB

# re-measure S3DF cache size
sys.getsizeof(ts_s3df) / BYTES_IN_GB
asizeof(ts_s3df) / BYTES_IN_GB


ts_df_0_excl_tDelta = ts_df_0[cols_excl_tDelta]
ts_df_excl_tDelta = ts_df[cols_excl_tDelta]

comp_excl_tDelta = \
    ((ts_df_0_excl_tDelta == ts_df_excl_tDelta) |
     (pandas.isnull(ts_df_0_excl_tDelta) &
      pandas.isnull(ts_df_excl_tDelta))).all()
assert comp_excl_tDelta.all()

del ts_df_0_excl_tDelta
del ts_df_excl_tDelta


ts_df_0_tDelta = ts_df_0[_T_DELTA_COL]
ts_df_tDelta = ts_df[_T_DELTA_COL]

_non_nan_chks = \
    pandas.notnull(ts_df_0_tDelta) & \
    pandas.notnull(ts_df_tDelta)
assert numpy.allclose(
    ts_df_0_tDelta.loc[_non_nan_chks],
    ts_df_tDelta.loc[_non_nan_chks],
    equal_nan=False)

del ts_df_0_tDelta
del ts_df_tDelta


prep_ts_adf, orig_to_prep_col_map = \
    ts_adf.prep(
        forceCat=bp.params.data.timeser_force_cat,
        forceCatIncl=bp.params.data.timeser_force_cat_incl,
        forceCatExcl=bp.params.data.timeser_force_cat_excl,

        forceNum=bp.params.data.timeser_force_num,
        forceNumIncl=bp.params.data.timeser_force_num_incl,
        forceNumExcl=bp.params.data.timeser_force_num_excl,

        fill=dict(
            method=bp.params.data.timeser_num_null_fill_method,
            value=None,
            outlierTails=bp.params.data.timeser_num_outlier_tails,
            fillOutliers=False),

        scaler=bp.params.data.num_data_scaler,

        assembleVec=None,

        loadPath=None,
        savePath=TS_ADF_DATA_TRANSFORM_PATH,

        returnOrigToPrepColMap=True,
        verbose=True)
