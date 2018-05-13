import arimo.debug

from data import DATASET_NAMES, DISP_CASE_EQ_GEN_TYPE_NAME
from PredMaintProject_n_PPPBlueprints import PRED_MAINT_PROJECT


arimo.debug.ON = True


N_SAMPLES = 1000000


files_ts_adf = \
    PRED_MAINT_PROJECT.load_equipment_data(
        equipment_instance_id_or_data_set_name=DATASET_NAMES[DISP_CASE_EQ_GEN_TYPE_NAME],
        _files_based=True, _spark=True,
        reprSampleSize=N_SAMPLES,
        verbose=True)

files_ts_adf.reprSample   # FAST


files_ts_adf_select_all = files_ts_adf.copy()

files_ts_adf_select_all.reprSample   # SLOW


ts_adf = PRED_MAINT_PROJECT.load_equipment_data(
    equipment_instance_id_or_data_set_name=DATASET_NAMES[DISP_CASE_EQ_GEN_TYPE_NAME],
    _files_based=False, _spark=True,
    reprSampleSize=N_SAMPLES,
    verbose=True)

ts_adf.reprSample   # SLOWEST
