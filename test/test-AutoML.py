from __future__ import print_function

import argparse
import joblib
import os
from sklearn.model_selection import train_test_split

from arimo.blueprints.core.CrossSect import Regr
from arimo.util import cache, pkl
import arimo.debug

from PanasonicColdChain import _S3_CLIENT, S3_BUCKET_NAME, S3_CACHE_PREFIX, LOCAL_CACHE_DIR_PATH
from PanasonicColdChain.feature_eng.refrig_grp import refrig_df


STORE_ID = '1004'
EQUIPMENT_ID = 'D38'


FROM_DATE_STR = '2014-01-10'
TO_DATE_STR = '2014-12-31'


TEST_PROPORTION = .32
RAND_SEED = 1


TRAIN_FILE_NAME = 'PPP-Train{}.{}'.format(pkl.PKL_EXT, pkl.COMPAT_COMPRESS)

LOCAL_TRAIN_FILE_PATH = \
    os.path.join(
        LOCAL_CACHE_DIR_PATH,
        TRAIN_FILE_NAME)

S3_TRAIN_FILE_KEY = \
    os.path.join(
        S3_CACHE_PREFIX,
        TRAIN_FILE_NAME)


TEST_FILE_NAME = 'PPP-Test{}.{}'.format(pkl.PKL_EXT, pkl.COMPAT_COMPRESS)

LOCAL_TEST_FILE_PATH = \
    os.path.join(
        LOCAL_CACHE_DIR_PATH,
        TEST_FILE_NAME)

S3_TEST_FILE_KEY = \
    os.path.join(
        S3_CACHE_PREFIX,
        TEST_FILE_NAME)


LABEL_VAR = 'discharge_temperature'

CAT_PRED_VARS = [
    'alarm1',
    'alarm2',
    'alarm3',
    'alarm4',
    'compressor1',
    'compressor2',
    'compressor3',
    'compressor4',
    'ice_thermal_storage_input',
    'low_pressure_control_cycle',
    'operation_mode']

NUM_PRED_VARS = [
    '__avg_setting_temp__',
    '__n_disp_cases__',
    'differential_pressure_value_under_energy_saving_control',
    'high_pressure',
    'judgment_reference_temperature',
    'low_pressure',
    'low_pressure_1',
    'low_pressure_value_for_chainging_to_off_operation',
    'low_pressure_value_for_chainging_to_on_operation',
    'lower_limit_of_differential_btw_high_and_low_pressure',
    'lower_limit_of_low_pressure',
    'standard_value_of_low_pressure_for_chainging_to_off_operation',
    'standard_value_of_low_pressure_for_chainging_to_on_operation',
    'suction_temperature',
    'the_amount_of_change_in_pressure_under_low_pressure_control',
    'upper_limit_of_low_pressure']

PRED_VARS = CAT_PRED_VARS + NUM_PRED_VARS


TIME_BUDGET_IN_MINS = 6

_argparser = argparse.ArgumentParser()
_argparser.add_argument('--mins', default=TIME_BUDGET_IN_MINS, type=int)
_args = _argparser.parse_args()


@cache.S3CacheDecor(
    s3_client=_S3_CLIENT,
    s3_bucket=S3_BUCKET_NAME,
    s3_cache_dir_prefix=S3_CACHE_PREFIX,
    local_cache_dir_path=LOCAL_CACHE_DIR_PATH,
    serializer='joblib',
    file_name_lambda=lambda **kwargs: TRAIN_FILE_NAME,
    pre_condition_lambda=None,
    validation_lambda=None,
    post_process_lambda=None,
    verbose=True)
def train_df(_cache_verbose=True):
    df = refrig_df(
        store_id=STORE_ID,
        refrig_id=EQUIPMENT_ID,
        from_date_str=FROM_DATE_STR,
        to_date_str=TO_DATE_STR)

    trn_df, tst_df = \
        train_test_split(
            df,
            test_size=TEST_PROPORTION,
            random_state=RAND_SEED)

    joblib.dump(
        tst_df,
        filename=LOCAL_TEST_FILE_PATH,
        compress=(pkl.COMPAT_COMPRESS, pkl.MAX_COMPRESS_LVL),
        protocol=pkl.COMPAT_PROTOCOL)

    _S3_CLIENT.upload_file(
        Filename=LOCAL_TEST_FILE_PATH,
        Bucket=S3_BUCKET_NAME,
        Key=S3_TEST_FILE_KEY)

    return trn_df


@cache.S3CacheDecor(
    s3_client=_S3_CLIENT,
    s3_bucket=S3_BUCKET_NAME,
    s3_cache_dir_prefix=S3_CACHE_PREFIX,
    local_cache_dir_path=LOCAL_CACHE_DIR_PATH,
    serializer='joblib',
    file_name_lambda=lambda **kwargs: TEST_FILE_NAME,
    pre_condition_lambda=None,
    validation_lambda=None,
    post_process_lambda=None,
    verbose=True)
def test_df(_cache_verbose=True):
    df = refrig_df(
        store_id=STORE_ID,
        refrig_id=EQUIPMENT_ID,
        from_date_str=FROM_DATE_STR,
        to_date_str=TO_DATE_STR)

    trn_df, tst_df = \
        train_test_split(
            df,
            test_size=TEST_PROPORTION,
            random_state=RAND_SEED)

    joblib.dump(
        trn_df,
        filename=LOCAL_TRAIN_FILE_PATH,
        compress=(pkl.COMPAT_COMPRESS, pkl.MAX_COMPRESS_LVL),
        protocol=pkl.COMPAT_PROTOCOL)

    _S3_CLIENT.upload_file(
        Filename=LOCAL_TRAIN_FILE_PATH,
        Bucket=S3_BUCKET_NAME,
        Key=S3_TRAIN_FILE_KEY)

    return trn_df


arimo.debug.ON = True


bp = Regr.AutoMLBlueprint(
    params={
        'data.label.var': LABEL_VAR,
        'data.pred_vars': PRED_VARS,
        'data.force_cat': CAT_PRED_VARS,
        'data.force_num': NUM_PRED_VARS,
        'model.factory.time_left_for_this_task': _args.mins * 60,
        'model.factory.per_run_time_limit': _args.mins * 60,
        'model.train.objective': 'mean_absolute_error'})

bp.train(__df__=train_df())

print(bp.eval(
    __df__=test_df(),
    __excl_outliers__=True))
