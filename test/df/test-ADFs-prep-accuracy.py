import numpy

from arimo.IoT.PredMaint import project


PROJECT = project('PanaAP-CC')


EQ_GEN_TP = 'refrig'
EQ_UNQ_TP_GRP = 'refrigerator_general'

LABEL_VARS = 'high_pressure', 'low_pressure', 'discharge_temperature', 'suction_temperature'

DATE = '2016-07-01'


ppp_bp = \
    PROJECT._ppp_blueprint(
        equipment_general_type_name=EQ_GEN_TP, equipment_unique_type_group_name=EQ_UNQ_TP_GRP,
        timeser_input_len=3, incl_time_features=True, excl_mth_time_features=False,
        verbose=True)

for label_var in LABEL_VARS:
    ppp_bp.params.model.component_blueprints[label_var].model.ver = '_'   # so not empty


arrow_adf = \
    PROJECT.load_equipment_data(
        '{}---{}'.format(EQ_GEN_TP.upper(), EQ_UNQ_TP_GRP),
        _from_files=True, _spark=False,
        verbose=True)


prep_arrow_adfs = \
    ppp_bp.prep_data(
        df=arrow_adf,
        __mode__='train',
        verbose=True)


df_from_spark = \
    ppp_bp.prep_data(
        df=PROJECT.load_equipment_data(
            '{}---{}'.format(EQ_GEN_TP.upper(), EQ_UNQ_TP_GRP),
            _from_files=True, _spark=True,
            verbose=True),
        __mode__='score',
        __vectorize__=True,
        verbose=True) \
    .filterByPartitionKeys(('date', DATE)) \
    .collect() \
    .sort_values(
        by=['equipment_instance_id', 'date_time'],
        axis='index',
        ascending=True,
        kind='quicksort',
        na_position='last',
        inplace=False)


Xs_from_arrow = {}
Xs_from_spark = {}

for label_var in LABEL_VARS:
    component_bp_data_params = \
        ppp_bp.params.model.component_blueprints[label_var].data

    Xs_from_arrow[label_var] = \
        X_from_arrow = \
        prep_arrow_adfs[label_var] \
            .filterByPartitionKeys(('date', DATE)) \
            .collect() \
            [list(component_bp_data_params._cat_prep_cols + component_bp_data_params._num_prep_cols)] \
            .values

    Xs_from_spark[label_var] = \
        X_from_spark = \
        numpy.vstack(
            v.toArray()
            for v in df_from_spark['__Xvec__' + label_var])

    assert numpy.allclose(
            X_from_arrow,
            X_from_spark)
