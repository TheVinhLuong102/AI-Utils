import numpy

from arimo.df.spark_on_files import FileADF

from PanasonicColdChain import _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY
from PanasonicColdChain.PredMaint.PPPAD import PRED_MAINT_PROJECT


EQUIPMENT_GENERAL_TYPE = 'refrig'
EQUIPMENT_UNIQUE_TYPE = 'inverter_2_multi_comp_refrigerator'

EQUIPMENT_INSTANCE_ID = '1018_c34'

BLUEPRINT_UUID = '7a57137a-ef01-4c77-9362-54aa2a55bcf2'

LABEL_VARS = 'low_pressure', 'high_pressure', 'suction_temperature'

DATE = '2016-07-01'


bp = PRED_MAINT_PROJECT._ppp_blueprint(
    uuid=BLUEPRINT_UUID,
    verbose=False)


fadf0 = PRED_MAINT_PROJECT.load_equipment_data(EQUIPMENT_INSTANCE_ID, iCol=None)

fadf1 = fadf0.filterByPartitionKeys(('date', DATE))

if '__disp_cases__' in fadf1.columns:
    path = fadf1.path + '---FIXED'

    fadf1.drop('__disp_cases__') \
        .save(path=path,
              aws_access_key_id=_AWS_ACCESS_KEY_ID,
              aws_secret_access_key=_AWS_SECRET_ACCESS_KEY)

    fadf1 = FileADF(
        path=path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID,
        aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        tCol='date_time')


score_prep_fadf1 = \
    bp.prep_data(
        df=fadf1,
        __mode__='score',
        __vectorize__=True)

score_prep_df1_from_spark = \
    score_prep_fadf1.toPandas()

score_prep_df1 = \
    score_prep_fadf1._piecePandasDF(
        list(score_prep_fadf1.pieceSubPaths)[0])


Xs_from_spark = {}
Xs_from_pandas = {}

for label_var in LABEL_VARS:
    Xs_from_spark[label_var] = \
        X_from_spark = \
            numpy.vstack(
                v.toArray()
                for v in score_prep_df1_from_spark['__Xvec__' + label_var])
    
    component_bp_data_params = bp.params.model.component_blueprints[label_var].data

    Xs_from_pandas[label_var] = \
        X_from_pandas = \
            score_prep_df1[
                list(component_bp_data_params._cat_prep_cols +
                     component_bp_data_params._num_prep_cols)].values

    assert numpy.allclose(
        X_from_spark,
        X_from_pandas,
        atol=1e-4)
