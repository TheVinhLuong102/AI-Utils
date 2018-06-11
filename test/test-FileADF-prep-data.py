import numpy

from arimo.df.spark_from_files import ArrowSparkADF
from arimo.IoT.PredMaint import project


PROJECT = project('PanaAP-CC')


EQUIPMENT_GENERAL_TYPE = 'refrig'
EQUIPMENT_UNIQUE_TYPE = 'inverter_2_multi_comp_refrigerator'

EQUIPMENT_INSTANCE_ID = '1018_c34'

BLUEPRINT_UUID = 'REFRIG---inverter_2_multi_comp_refrigerator---to-2015-12---20e7dfd0-dcfb-4ec0-b597-def65ecca5c5'

LABEL_VARS = 'low_pressure', 'high_pressure', 'suction_temperature'

DATE = '2016-07-01'


bp = PROJECT._ppp_blueprint(
    uuid=BLUEPRINT_UUID,
    verbose=False)


fadf0 = PROJECT.load_equipment_data(EQUIPMENT_INSTANCE_ID, iCol=None, _spark=True)

fadf1 = fadf0.filterByPartitionKeys(('date', DATE))

if '__disp_cases__' in fadf1.columns:
    path = fadf1.path + '---FIXED'
    fadf1.drop('__disp_cases__') \
        .save(path=path,
              aws_access_key_id=PROJECT.params.s3.access_key_id,
              aws_secret_access_key=PROJECT.params.s3.secret_access_key)
    fadf1 = ArrowSparkADF(
        path=path,
        aws_access_key_id=PROJECT.params.s3.access_key_id,
        aws_secret_access_key=PROJECT.params.s3.secret_access_key,
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


df = fadf1.toPandas()