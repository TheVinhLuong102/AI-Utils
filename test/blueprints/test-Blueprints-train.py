from __future__ import print_function

import argparse
import os

from arimo.blueprints.base import load
from arimo.IoT.PredMaint import project, _PARQUET_EXT


PROJECT = project('PanaAP-CC')

PPP_BP_UUID = 'DISP_CASE---business_freezer---to-2018-01---8cad4273-d93f-414a-84fb-64cc583c55c5'
SUP_BP_UUID = 'DISP_CASE---business_freezer---to-2018-01---8cad4273-d93f-414a-84fb-64cc583c55c5---inside_temperature---7778ece4-33ae-4e38-9c1e-a8a2fb19020a'

DATASET_NAME = 'DISP_CASE---business_freezer'

DATA_PATH = os.path.join(PROJECT.params.s3.equipment_data.dir_path, DATASET_NAME + _PARQUET_EXT)

LABEL_VAR = 'inside_temperature'


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--ppp', action='store_true')
arg_parser.add_argument('--incr', action='store_true')
arg_parser.add_argument('--arrow', action='store_true')
arg_parser.add_argument('--spark', action='store_true')
args = arg_parser.parse_args()


ppp_bp = PROJECT._ppp_blueprint(uuid=PPP_BP_UUID)

if args.ppp:
    bp = ppp_bp

else:
    bp = load(
        s3_bucket=PROJECT.params.s3.bucket,
        s3_dir_prefix=os.path.join(
            PROJECT.params.s3.ppp.blueprints_dir_prefix,
            SUP_BP_UUID),
        aws_access_key_id=PROJECT.params.s3.access_key_id,
        aws_secret_access_key=PROJECT.params.s3.secret_access_key,
        s3_client=PROJECT.s3_client,
        verbose=False)

    if bp.params.model.ver:
        assert bp.params.model.ver == ppp_bp.params.model.component_blueprints[LABEL_VAR].model.ver, \
            '*** {} vs. {} ***'.format(bp.params.model.ver, ppp_bp.params.model.component_blueprints[LABEL_VAR].model.ver)

    bp.params.model.ver = \
        ppp_bp.params.model.component_blueprints[LABEL_VAR].model.ver \
        if args.incr \
        else None


df = (PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=True, _spark=True,
        set_i_col=False, set_t_col=True)
      if args.spark
      else PROJECT.load_equipment_data(
            DATASET_NAME,
            _from_files=True, _spark=False,
            set_i_col=False, set_t_col=True)) \
    if args.arrow \
    else (PROJECT.load_equipment_data(
            DATASET_NAME,
            _from_files=False, _spark=True,
            set_i_col=False, set_t_col=True)
          if args.spark
          else DATA_PATH)


print('\n*** {} .train(df= {} ) ***\n'.format(bp, df))


bp.train(
    df=df,
    aws_access_key_id=PROJECT.params.s3.access_key_id,
    aws_secret_access_key=PROJECT.params.s3.secret_access_key)
