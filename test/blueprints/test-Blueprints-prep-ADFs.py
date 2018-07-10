from __future__ import print_function

import os

from arimo.blueprints.base import load
from arimo.IoT.PredMaint import project


PROJECT = project('PanaAP-CC')

PPP_BP_UUID = 'DISP_CASE---business_freezer---to-2018-01---61b944fa-8faf-4683-8bac-a90eaec0e41f'
LABELED_BP_UUID = 'DISP_CASE---business_freezer---to-2018-01---61b944fa-8faf-4683-8bac-a90eaec0e41f---inside_temperature---2ef0937e-29f8-4300-8db2-f36bb58d6002'

DATASET_NAME = 'DISP_CASE---business_freezer'

LABEL_VAR = 'inside_temperature'


ppp_bp = PROJECT._ppp_blueprint(uuid=PPP_BP_UUID)
print(ppp_bp)

labeled_bp = \
    load(s3_bucket=PROJECT.params.s3.bucket,
         s3_dir_prefix=os.path.join(
            PROJECT.params.s3.ppp.blueprints_dir_prefix,
            LABELED_BP_UUID),
         s3_client=PROJECT.s3_client,
         verbose=False)
print(labeled_bp)


arrow_adf = \
    PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=True, _spark=False,
        set_i_col=True, set_t_col=True)
print(arrow_adf)


spark_adf = \
    PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=False, _spark=True,
        set_i_col=True, set_t_col=True)
print(spark_adf)


arrow_spark_adf = \
    PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=True, _spark=True,
        set_i_col=True, set_t_col=True)
print(arrow_spark_adf)


ppp_prep_arrow_adf__train = \
    ppp_bp.prep_data(
        df=arrow_adf,
        __mode__='train')[LABEL_VAR]   # count & sampling will be triggered because of adf.suffNonNull(label_var_name)
print('PPP Blueprint-prepped Arrow ADF for Train: {}'.format(ppp_prep_arrow_adf__train))


ppp_prep_arrow_adf__eval = \
    ppp_bp.prep_data(
        df=arrow_adf,
        __mode__='score')[LABEL_VAR]

# print('PPP Blueprint-prepped Arrow ADF for Eval: {}'.format(ppp_prep_arrow_adf__eval))


# ppp_prep_arrow_adf__score = \
#     ppp_bp.prep_data(
#         df=arrow_adf,
#         __mode__='eval')[LABEL_VAR]

# print('PPP Blueprint-prepped Arrow ADF for Score: {}'.format(ppp_prep_arrow_adf__score))



