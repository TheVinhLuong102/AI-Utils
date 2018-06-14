import numpy
import tqdm

from arimo.IoT.PredMaint import project


PROJECT = project('PanaAP-CC')


EQ_GEN_TP = 'refrig'   # 'disp_case'

if EQ_GEN_TP == 'disp_case':
    EQ_UNQ_TP_GRP = 'ex_display_case'
    LABEL_VAR = 'inside_temperature'
else:
    EQ_UNQ_TP_GRP = 'inverter_4_multi_comp_refrigerator'
    LABEL_VAR = 'suction_temperature'

SEQ_LEN = 1   # 9

CHUNK_SIZE = 10 ** 4

BATCH_SIZE = 10 ** 3

N_BATCHES = 10 ** 3

N_THREADS = 4


ppp_bp = PROJECT._ppp_blueprint(
    equipment_general_type_name=EQ_GEN_TP,
    equipment_unique_type_group_name=EQ_UNQ_TP_GRP,
    incl_time_features=True,
    timeser_input_len=SEQ_LEN)


arrow_adf = PROJECT.load_equipment_data(
    '{}---{}'.format(EQ_GEN_TP.upper(), EQ_UNQ_TP_GRP),
    _on_files=True, _spark=False,
    set_i_col=SEQ_LEN > 1)


prep_arrow_adf = ppp_bp.prep_data(df=arrow_adf, __mode__='train')[LABEL_VAR]

component_bp_params = ppp_bp.params.model.component_blueprints[LABEL_VAR]

feature_cols = component_bp_params.data._cat_prep_cols + component_bp_params.data._num_prep_cols


# *** ArrowADF.gen(...) ***
gen_instance = \
    prep_arrow_adf.gen(
        feature_cols +
        ((- component_bp_params.pred_horizon_len - component_bp_params.max_input_ser_len + 1,
          - component_bp_params.pred_horizon_len)
         if SEQ_LEN > 1
         else ()),
        component_bp_params.data.label.var,
        n=BATCH_SIZE,
        sampleN=CHUNK_SIZE,
        withReplacement=False,
        seed=None,
        anon=True,
        collect='numpy',
        pad=numpy.nan,
        filter={component_bp_params.data.label.var: (-100, 100)},
        nThreads=N_THREADS)

print(gen_instance)
print(gen_instance.colsLists)


g0 = gen_instance()

for _ in tqdm.tqdm(range(N_BATCHES)):
    x, y = next(g0)


# *** S3ParquetDataset(Queue)Reader.generate_chunk(...) ***
cross_sect_dldf = \
    prep_arrow_adf._CrossSectDLDF(
        feature_cols=feature_cols,
        target_col=component_bp_params.data.label.var,
        n=BATCH_SIZE,
        sampleN=CHUNK_SIZE,
        filter={component_bp_params.data.label.var: (-100, 100)},
        nThreads=N_THREADS)

print(cross_sect_dldf)


g1 = cross_sect_dldf.generate_chunk()

for _ in tqdm.tqdm(range(N_BATCHES * BATCH_SIZE // CHUNK_SIZE)):
    chunk_df = next(g1)
