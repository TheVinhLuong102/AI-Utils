from __future__ import print_function

import argparse
import os
import tqdm

from arimo.blueprints import validate_blueprint_params
from arimo.util.iterables import flatten
import arimo.debug

from PanasonicColdChain import _AWS_ACCESS_KEY_ID, _AWS_SECRET_ACCESS_KEY

from data import DATASET_NAMES, REFRIG_EQ_GEN_TYPE_NAME, DISP_CASE_EQ_GEN_TYPE_NAME
from PredMaintProject_n_PPPBlueprints import PRED_MAINT_PROJECT, PPP_BLUEPRINTS


PARALLELISM = 68

TIME_SER_LEN = 9

MIN_N_PIECES = 1

N_SAMPLES = 10 ** 6

CHUNK_SIZE = 10 ** 5

BATCH_SIZE = 10 ** 3


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--refrig', action='store_true')
arg_parser.add_argument('--parallelism', default=PARALLELISM)
arg_parser.add_argument('--scaler', default='standard')
arg_parser.add_argument('--vectorize', action='store_true')
arg_parser.add_argument('--time-ser-len', default=TIME_SER_LEN)
arg_parser.add_argument('--n-samples', default=N_SAMPLES)
arg_parser.add_argument('--batch', default=BATCH_SIZE)
arg_parser.add_argument('--debug', action='store_true')

args = arg_parser.parse_args()

if args.debug:
    arimo.debug.ON = True

arimo.debug.ON = True


# args.refrig = True

dataset_name = \
    REFRIG_EQ_GEN_TYPE_NAME \
    if args.refrig \
    else DISP_CASE_EQ_GEN_TYPE_NAME

dataset_path = \
    os.path.join(
        PRED_MAINT_PROJECT.params.s3.equipment_data_dir_path,
        '{}.parquet'.format(DATASET_NAMES[dataset_name]))


ts_fadf = PRED_MAINT_PROJECT.load_equipment_data(DATASET_NAMES[dataset_name])
ts_fadf.iCol = None
ts_fadf.reprSampleNPieces = 1

prep_ts_fadf = ts_fadf.prep()
prep_ts_fadf

pieceSubPath = list(prep_ts_fadf.pieceSubPaths)[0]

df = prep_ts_fadf._piecePandasDF(pieceSubPath)

ts_ppp_bp = PPP_BLUEPRINTS[dataset_name]
ts_ppp_bp.params.data.num_data_scaler = args.scaler
ts_ppp_bp.params.data.id_col = None

train_prep_ts_fadfs = \
    ts_ppp_bp.prep_data(
        df=dataset_path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        reprSampleNPieces=MIN_N_PIECES,
        sparkConf={
            'spark.default.parallelism': args.parallelism,
            'spark.sql.shuffle.partitions': args.parallelism},
        __mode__='train',
        __vectorize__=False)


ts_ppp_bp.params.model.ver = ts_ppp_bp.model().ver

print(ts_ppp_bp.params.model.component_blueprints.values()[0].data)


vector_train_prep_ts_fadfs = \
    ts_ppp_bp.prep_data(
        df=dataset_path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        reprSampleNPieces=MIN_N_PIECES,
        __mode__='train',
        __vectorize__=True)


score_prep_ts_fadf = \
    ts_ppp_bp.prep_data(
        df=dataset_path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        reprSampleNPieces=MIN_N_PIECES,
        __mode__='score',
        __vectorize__=False)


vector_score_prep_ts_fadf = \
    ts_ppp_bp.prep_data(
        df=dataset_path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        reprSampleNPieces=MIN_N_PIECES,
        __mode__='score',
        __vectorize__=True)


eval_prep_ts_fadfs = \
    ts_ppp_bp.prep_data(
        df=dataset_path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        reprSampleNPieces=MIN_N_PIECES,
        __mode__='eval',
        __vectorize__=False)


vector_eval_prep_ts_fadfs = \
    ts_ppp_bp.prep_data(
        df=dataset_path,
        aws_access_key_id=_AWS_ACCESS_KEY_ID, aws_secret_access_key=_AWS_SECRET_ACCESS_KEY,
        reprSampleNPieces=MIN_N_PIECES,
        __mode__='eval',
        __vectorize__=True)


assert validate_blueprint_params(ts_ppp_bp)


LABEL_VAR = ts_ppp_bp.params.model.component_blueprints.keys()[0]
COMPONENT_BLUEPRINT_PARAMS = ts_ppp_bp.params.model.component_blueprints[LABEL_VAR]

argsForSampleGenNonVector = \
    LABEL_VAR, \
    COMPONENT_BLUEPRINT_PARAMS.data._cat_prep_cols + COMPONENT_BLUEPRINT_PARAMS.data._num_prep_cols + (-TIME_SER_LEN,)

argsForSampleGenNonVector = \
    LABEL_VAR, \
    COMPONENT_BLUEPRINT_PARAMS.data._cat_prep_cols + COMPONENT_BLUEPRINT_PARAMS.data._num_prep_cols


#argsForSampleGenVector = \
#    LABEL_VAR, \
#    (COMPONENT_BLUEPRINT_PARAMS.data._prep_vec_col, -TIME_SER_LEN)





train_prep_ts_fadf = train_prep_ts_fadfs[LABEL_VAR]


g = train_prep_ts_fadf.gen(*argsForSampleGenNonVector, anon=True, n=30000)
# a = next(g)

for i in tqdm.tqdm(range(100000)):
    _ = next(g)



pieceSubPath = list(adf.pieceSubPaths)[0]

df = adf._piecePandasDF(pieceSubPath)


pieceSubPath = list(vector_score_prep_ts_fadf.pieceSubPaths)[0]

piecePandasDF = vector_score_prep_ts_fadf._piecePandasDF(pieceSubPath)

arrowTable = vector_train_prep_ts_fadf._pieceArrowTable(pieceSubPath)

batches = arrowTable.to_batches(chunksize=CHUNK_SIZE)
batch = batches[0]

batchDF = batch.to_pandas(nthreads=4)

for pieceSubPath in tqdm.tqdm(vector_train_prep_ts_fadf.pieceSubPaths):
    df = vector_train_prep_ts_fadf._piecePandasDF(pieceSubPath)






prepare_vector_overtime_train_prep_ts_fadf = \
    vector_train_prep_ts_fadfs[LABEL_VAR]._prepareArgsForSampleOrGenOrPred(
        *argsForSampleGenVector,
        anon=True,
        n=100).adf




gen_vector_overtime_train_prep_ts_fadf = \
    vector_train_prep_ts_fadfs[LABEL_VAR].gen(
        *argsForSampleGenVector,
        n=100, sampleN=N_SAMPLES,
        anon=True)

a = next(gen_vector_overtime_train_prep_ts_fadf)


for i in tqdm.tqdm(range(1000000)):
    _ = next(gen_vector_overtime_train_prep_ts_fadf)




sample_non_vector_overtime_train_prep_ts_fadf = \
    train_prep_ts_fadfs[LABEL_VAR].sample(
        *argsForSampleGenNonVector,
        n=N_SAMPLES,
        maxNPieces=1)


sample_vector_overtime_train_prep_ts_fadf = \
    vector_train_prep_ts_fadfs[LABEL_VAR].sample(
        *argsForSampleGenVector,
        n=N_SAMPLES,
        maxNPieces=1)


gen_vector_overtime_train_prep_ts_fadf = \
    vector_train_prep_ts_fadfs[LABEL_VAR].gen(
        *argsForSampleGenVector,
        anon=True,
        n=100)


for i in tqdm.tqdm(range(1000000)):
    _ = next(gen_vector_overtime_train_prep_ts_fadf)



g = prepare_vector_overtime_train_prep_ts_fadf.toLocalIterator()


# sample_non_vector_overtime_train_prep_ts_fadf.cache()    # 30 mins for 1 mil

gen_sample_non_vector_overtime_train_prep_ts_fadf = \
    vector_train_prep_ts_fadfs[LABEL_VAR].sample(
        *argsForSampleGenVector,
        n=N_SAMPLES,
        maxNPieces=1)

for i in tqdm.tqdm(range(N_SAMPLES)):
    _ = next(gen_sample_non_vector_overtime_train_prep_ts_fadf)



sample_vector_overtime_train_prep_ts_fadf.cache()

gen_vector_overtime_train_prep_ts_fadf = \
    sample_vector_overtime_train_prep_ts_fadf.toLocalIterator()

for i in tqdm.tqdm(range(N_SAMPLES)):
    _ = next(gen_vector_overtime_train_prep_ts_fadf)


import numpy
import itertools

rows = list(itertools.islice(gen_vector_overtime_train_prep_ts_fadf, 10))


sample_vector_overtime_train_prep_ts_fadf._cache.colWidth[sample_vector_overtime_train_prep_ts_fadf.columns[1]] = 74

c = sample_vector_overtime_train_prep_ts_fadf._collectCols(
    rows,
    cols=sample_vector_overtime_train_prep_ts_fadf.columns[1:],
    asPandas=False,
    overTime=True,
    padUpToNTimeSteps=20,
    padValue=.1303,
    padBefore=True)




