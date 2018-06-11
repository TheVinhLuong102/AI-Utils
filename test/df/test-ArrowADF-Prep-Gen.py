import numpy
import tqdm

from arimo.IoT.PredMaint import project


PROJECT = project('PanaAP-CC')


EQ_GEN_TP = 'disp_case'
EQ_UNQ_TP_GRP = 'ex_display_case'

LABEL_VAR = 'inside_temperature'

SEQ_LEN = 9

BATCH_SIZE = 10 ** 3

N_BATCHES = 10 ** 3

N_THREADS = 4


ppp_bp = PROJECT._ppp_blueprint(
    equipment_general_type_name=EQ_GEN_TP,
    equipment_unique_type_group_name=EQ_UNQ_TP_GRP,
    timeser_input_len=SEQ_LEN)


arrow_adf = PROJECT.load_equipment_data(
    '{}---{}'.format(EQ_GEN_TP.upper(), EQ_UNQ_TP_GRP),
    _on_files=True, _spark=False,
    iCol=SEQ_LEN > 1)


prep_arrow_adf = ppp_bp.prep_data(df=arrow_adf, __mode__='train')[LABEL_VAR]

component_bp_params = ppp_bp.params.model.component_blueprints[LABEL_VAR]

g = prep_arrow_adf.gen(
    component_bp_params.data._cat_prep_cols + component_bp_params.data._num_prep_cols +
    ((- component_bp_params.pred_horizon_len - component_bp_params.max_input_ser_len + 1,
      - component_bp_params.pred_horizon_len)
     if SEQ_LEN > 1
     else ()),
    component_bp_params.data.label.var,
    n=BATCH_SIZE,
    sampleN=10 ** (4 if SEQ_LEN > 1 else 5),
    withReplacement=False,
    seed=None,
    anon=True,
    collect='numpy',
    pad=numpy.nan,
    filter={},
    n_threads=N_THREADS)()


for _ in tqdm.tqdm(range(N_BATCHES)):
    x, y = next(g)
