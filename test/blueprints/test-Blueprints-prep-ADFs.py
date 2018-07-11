from __future__ import print_function

import os

from arimo.df import _ADFABC
_ADFABC._DEFAULT_REPR_SAMPLE_SIZE = 10 ** 3

from arimo.df.from_files import ArrowADF, _ArrowADFABC
ArrowADF._DEFAULT_KWARGS['reprSampleMinNPieces'] = ArrowADF._REPR_SAMPLE_MIN_N_PIECES = \
    _ArrowADFABC._REPR_SAMPLE_MIN_N_PIECES = 1
ArrowADF._DEFAULT_KWARGS['reprSampleSize'] = ArrowADF._DEFAULT_REPR_SAMPLE_SIZE = 1

from arimo.df.spark import SparkADF
SparkADF._DEFAULT_KWARGS['reprSampleSize'] = SparkADF._DEFAULT_REPR_SAMPLE_SIZE = 10 ** 3

from arimo.df.spark_from_files import ArrowSparkADF
ArrowSparkADF._DEFAULT_KWARGS['reprSampleMinNPieces'] = ArrowSparkADF._REPR_SAMPLE_MIN_N_PIECES = 1
ArrowSparkADF._DEFAULT_KWARGS['reprSampleSize'] = ArrowSparkADF._DEFAULT_REPR_SAMPLE_SIZE = 10 ** 3

from arimo.blueprints.base import load
from arimo.IoT.PredMaint import project, _PARQUET_EXT


PROJECT = project('PanaAP-CC')

PPP_BP_UUID = 'DISP_CASE---business_freezer---to-2018-01---8cad4273-d93f-414a-84fb-64cc583c55c5'
SUP_BP_UUID = 'DISP_CASE---business_freezer---to-2018-01---8cad4273-d93f-414a-84fb-64cc583c55c5---inside_temperature---7778ece4-33ae-4e38-9c1e-a8a2fb19020a'

DATASET_NAME = 'DISP_CASE---business_freezer'

DATA_PATH = os.path.join(PROJECT.params.s3.equipment_data.dir_path, DATASET_NAME + _PARQUET_EXT)
AWS_ACCESS_KEY_ID = PROJECT.params.s3.access_key_id
AWS_SECRET_ACCESS_KEY = PROJECT.params.s3.secret_access_key

LABEL_VAR = 'inside_temperature'


ppp_bp = PROJECT._ppp_blueprint(uuid=PPP_BP_UUID)
print(ppp_bp)

sup_bp = \
    load(s3_bucket=PROJECT.params.s3.bucket,
         s3_dir_prefix=os.path.join(
            PROJECT.params.s3.ppp.blueprints_dir_prefix,
            SUP_BP_UUID),
         aws_access_key_id=PROJECT.params.s3.access_key_id,
         aws_secret_access_key=PROJECT.params.s3.secret_access_key,
         s3_client=PROJECT.s3_client,
         verbose=False)
assert sup_bp.params.model.ver is None, \
    '*** {} ***'.format(sup_bp.params.model.ver)
print(sup_bp)


arrow_adf = \
    PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=True, _spark=False,
        set_i_col=False, set_t_col=True)
print(arrow_adf)


spark_adf = \
    PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=False, _spark=True,
        set_i_col=False, set_t_col=True)
print(spark_adf)


arrow_spark_adf = \
    PROJECT.load_equipment_data(
        DATASET_NAME,
        _from_files=True, _spark=True,
        set_i_col=False, set_t_col=True)
print(arrow_spark_adf)


# 1
ppp_prep_data_path__train = \
    ppp_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='train',
        __vectorize__=False)[LABEL_VAR]   # counting & sampling will be triggered because of adf.suffNonNull(...)
print('PPPBlueprint-prepped Data Path for Training:\n{}\n'
    .format(ppp_prep_data_path__train))


# 2
ppp_prep_data_path__train__vectorized = \
    ppp_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='train',
        __vectorize__=True)[LABEL_VAR]   # counting & sampling will be triggered because of adf.suffNonNull(...)
print('PPPBlueprint-prepped Data Path for Training (Vectorized):\n{}\n'
    .format(ppp_prep_data_path__train__vectorized))


# 3
ppp_prep_data_path__score = \
    ppp_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='score',
        __vectorize__=False)
print('PPPBlueprint-prepped Data Path for Scoring:\n{}\n'
    .format(ppp_prep_data_path__score))


# 4
ppp_prep_data_path__score__vectorized = \
    ppp_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='score',
        __vectorize__=True)
print('PPPBlueprint-prepped Data Path for Scoring (Vectorized):\n{}\n'
    .format(ppp_prep_data_path__score__vectorized))


# 5
ppp_prep_data_path__eval = \
    ppp_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='eval',
        __vectorize__=False)
print('PPPBlueprint-prepped Data Path for Eval:\n{}\n'
    .format(ppp_prep_data_path__eval))


# 6
ppp_prep_data_path__eval__vectorized = \
    ppp_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='eval',
        __vectorize__=True)
print('PPPBlueprint-prepped Data Path for Eval (Vectorized):\n{}\n'
    .format(ppp_prep_data_path__eval__vectorized))


# 7
ppp_prep_arrow_adf__train = \
    ppp_bp.prep_data(
        df=arrow_adf,
        __mode__='train')[LABEL_VAR]   # counting & sampling will be triggered because of adf.suffNonNull(...)
print('PPPBlueprint-prepped ArrowADF for Training:\n{}\n({} --> {})\n\n(from {})\n'
    .format(ppp_prep_arrow_adf__train,
            ppp_prep_arrow_adf__train._mappers,
            ppp_prep_arrow_adf__train.sample(maxNPieces=1).columns.tolist(),
            arrow_adf))


# 8
ppp_prep_arrow_adf__score = \
    ppp_bp.prep_data(
        df=arrow_adf,
        __mode__='score')
print('PPPBlueprint-prepped ArrowADF for Scoring:\n{}\n({} --> {})\n\n(from {})\n'
    .format(ppp_prep_arrow_adf__score,
            ppp_prep_arrow_adf__score._mappers,
            ppp_prep_arrow_adf__score.sample(maxNPieces=1).columns.tolist(),
            arrow_adf))


# 9
ppp_prep_arrow_adf__eval = \
    ppp_bp.prep_data(
        df=arrow_adf,
        __mode__='eval')
print('PPPBlueprint-prepped ArrowADF for Eval:\n{}\n({} --> {})\n\n(from {})\n'
    .format(ppp_prep_arrow_adf__eval,
            ppp_prep_arrow_adf__eval._mappers,
            ppp_prep_arrow_adf__eval.sample(maxNPieces=1).columns.tolist(),
            arrow_adf))


# 10
ppp_prep_spark_adf__train = \
    ppp_bp.prep_data(
        df=spark_adf,
        __mode__='train',
        __vectorize__=False)[LABEL_VAR]   # counting & sampling will be triggered because of adf.suffNonNull(...)
print('PPPBlueprint-prepped SparkADF for Training:\n{}\n\n(from {})\n'
    .format(ppp_prep_spark_adf__train, spark_adf))


# 11
ppp_prep_spark_adf__train__vectorized = \
    ppp_bp.prep_data(
        df=spark_adf,
        __mode__='train',
        __vectorize__=True)[LABEL_VAR]   # counting & sampling will be triggered because of adf.suffNonNull(...)
print('PPPBlueprint-prepped SparkADF for Training (Vectorized):\n{}\n\n(from {})\n'
    .format(ppp_prep_spark_adf__train__vectorized, spark_adf))


# 12
ppp_prep_spark_adf__score = \
    ppp_bp.prep_data(
        df=spark_adf,
        __mode__='score',
        __vectorize__=False)
print('PPPBlueprint-prepped SparkADF for Scoring:\n{}\n\n(from {})\n'
    .format(ppp_prep_spark_adf__score, spark_adf))


# 13
ppp_prep_spark_adf__score__vectorized = \
    ppp_bp.prep_data(
        df=spark_adf,
        __mode__='score',
        __vectorize__=True)
print('PPPBlueprint-prepped SparkADF for Scoring (Vectorized):\n{}\n\n(from {})\n'
    .format(ppp_prep_spark_adf__score__vectorized, spark_adf))


# 14
ppp_prep_spark_adf__eval = \
    ppp_bp.prep_data(
        df=spark_adf,
        __mode__='eval',
        __vectorize__=False)
print('PPPBlueprint-prepped SparkADF for Eval:\n{}\n\n(from {})\n'
    .format(ppp_prep_spark_adf__eval, spark_adf))


# 15
ppp_prep_spark_adf__eval__vectorized = \
    ppp_bp.prep_data(
        df=spark_adf,
        __mode__='eval',
        __vectorize__=True)
print('PPPBlueprint-prepped SparkADF for Eval (Vectorized):\n{}\n\n(from {})\n'
    .format(ppp_prep_spark_adf__eval__vectorized, spark_adf))


# 16
ppp_prep_arrow_spark_adf__train = \
    ppp_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='train',
        __vectorize__=False)[LABEL_VAR]   # counting & sampling will be triggered because of adf.suffNonNull(...)
print('PPPBlueprint-prepped ArrowSparkADF for Training:\n{}\n\n(from {})\n'
    .format(ppp_prep_arrow_spark_adf__train, arrow_spark_adf))


# 17
ppp_prep_arrow_spark_adf__score = \
    ppp_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='score',
        __vectorize__=False)
print('PPPBlueprint-prepped ArrowSparkADF for Scoring:\n{}\n\n(from {})\n'
    .format(ppp_prep_arrow_spark_adf__score, arrow_spark_adf))


# 18
ppp_prep_arrow_spark_adf__score__vectorized = \
    ppp_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='score',
        __vectorize__=True)
print('PPPBlueprint-prepped ArrowSparkADF for Scoring (Vectorized):\n{}\n\n(from {})\n'
    .format(ppp_prep_arrow_spark_adf__score__vectorized, arrow_spark_adf))


# 19
ppp_prep_arrow_spark_adf__eval = \
    ppp_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='eval',
        __vectorize__=False)
print('PPPBlueprint-prepped ArrowSparkADF for Eval:\n{}\n\n(from {})\n'
    .format(ppp_prep_arrow_spark_adf__eval, arrow_spark_adf))


# 20
ppp_prep_arrow_spark_adf__eval__vectorized = \
    ppp_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='eval',
        __vectorize__=True)
print('PPPBlueprint-prepped ArrowSparkADF for Eval (Vectorized):\n{}\n\n(from {})\n'
    .format(ppp_prep_arrow_spark_adf__eval__vectorized, arrow_spark_adf))


# 21
sup_prep_data_path__train = \
    sup_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='train',
        __vectorize__=False)
print('SupervisedBlueprint-prepped Data Path for Training:\n{}\n'
    .format(sup_prep_data_path__train))


# 22
sup_prep_data_path__train__vectorized = \
    sup_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='train',
        __vectorize__=True)
print('SupervisedBlueprint-prepped Data Path for Training (Vectorized):\n{}\n'
    .format(sup_prep_data_path__train__vectorized))


# 23
sup_prep_data_path__score = \
    sup_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='score',
        __vectorize__=False)
print('SupervisedBlueprint-prepped Data Path for Scoring:\n{}\n'
    .format(sup_prep_data_path__score))


# 24
sup_prep_data_path__score__vectorized = \
    sup_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='score',
        __vectorize__=True)
print('SupervisedBlueprint-prepped Data Path for Scoring (Vectorized):\n{}\n'
    .format(sup_prep_data_path__score__vectorized))


# 25
sup_prep_data_path__eval = \
    sup_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='eval',
        __vectorize__=False)
print('SupervisedBlueprint-prepped Data Path for Eval:\n{}\n'
    .format(sup_prep_data_path__eval))


# 26
sup_prep_data_path__eval__vectorized = \
    sup_bp.prep_data(
        df=DATA_PATH,
        aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        __mode__='eval',
        __vectorize__=True)
print('SupervisedBlueprint-prepped Data Path for Eval (Vectorized):\n{}\n'
    .format(sup_prep_data_path__eval__vectorized))


# 27
sup_prep_arrow_adf__train = \
    sup_bp.prep_data(
        df=arrow_adf,
        __mode__='train')
print('SupervisedBlueprint-prepped ArrowADF for Training:\n{}\n({} --> {})\n\n(from {})\n'
    .format(sup_prep_arrow_adf__train,
            sup_prep_arrow_adf__train._mappers,
            sup_prep_arrow_adf__train.sample(maxNPieces=1).columns.tolist(),
            arrow_adf))


# 28
sup_prep_arrow_adf__score = \
    sup_bp.prep_data(
        df=arrow_adf,
        __mode__='score')
print('SupervisedBlueprint-prepped ArrowADF for Scoring:\n{}\n({} --> {})\n\n(from {})\n'
    .format(sup_prep_arrow_adf__score,
            sup_prep_arrow_adf__score._mappers,
            sup_prep_arrow_adf__score.sample(maxNPieces=1).columns.tolist(),
            arrow_adf))


# 29
sup_prep_arrow_adf__eval = \
    sup_bp.prep_data(
        df=arrow_adf,
        __mode__='eval')
print('SupervisedBlueprint-prepped ArrowADF for Eval:\n{}\n({} --> {})\n\n(from {})\n'
    .format(sup_prep_arrow_adf__eval,
            sup_prep_arrow_adf__eval._mappers,
            sup_prep_arrow_adf__eval.sample(maxNPieces=1).columns.tolist(),
            arrow_adf))


# 30
sup_prep_spark_adf__train = \
    sup_bp.prep_data(
        df=spark_adf,
        __mode__='train',
        __vectorize__=False)
print('SupervisedBlueprint-prepped SparkADF for Training:\n{}\n\n(from {})\n'
    .format(sup_prep_spark_adf__train, spark_adf))


# 31
sup_prep_spark_adf__train__vectorized = \
    sup_bp.prep_data(
        df=spark_adf,
        __mode__='train',
        __vectorize__=True)
print('SupervisedBlueprint-prepped SparkADF for Training (Vectorized):\n{}\n\n(from {})\n'
    .format(sup_prep_spark_adf__train__vectorized, spark_adf))


# 32
sup_prep_spark_adf__score = \
    sup_bp.prep_data(
        df=spark_adf,
        __mode__='score',
        __vectorize__=False)
print('SupervisedBlueprint-prepped SparkADF for Scoring:\n{}\n\n(from {})\n'
    .format(sup_prep_spark_adf__score, spark_adf))


# 33
sup_prep_spark_adf__score__vectorized = \
    sup_bp.prep_data(
        df=spark_adf,
        __mode__='score',
        __vectorize__=True)
print('SupervisedBlueprint-prepped SparkADF for Scoring (Vectorized):\n{}\n\n(from {})\n'
    .format(sup_prep_spark_adf__score__vectorized, spark_adf))


# 34
sup_prep_spark_adf__eval = \
    sup_bp.prep_data(
        df=spark_adf,
        __mode__='eval',
        __vectorize__=False)
print('SupervisedBlueprint-prepped SparkADF for Eval:\n{}\n\n(from {})\n'
    .format(sup_prep_spark_adf__eval, spark_adf))


# 35
sup_prep_spark_adf__eval__vectorized = \
    sup_bp.prep_data(
        df=spark_adf,
        __mode__='eval',
        __vectorize__=True)
print('SupervisedBlueprint-prepped SparkADF for Eval (Vectorized):\n{}\n\n(from {})\n'
    .format(sup_prep_spark_adf__eval__vectorized, spark_adf))


# 36
sup_prep_arrow_spark_adf__train = \
    sup_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='train')
print('SupervisedBlueprint-prepped ArrowSparkADF for Training:\n{}\n\n(from {})\n'
    .format(sup_prep_arrow_spark_adf__train, arrow_spark_adf))


# 37
sup_prep_arrow_spark_adf__score = \
    sup_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='score',
        __vectorize__=False)
print('SupervisedBlueprint-prepped ArrowSparkADF for Scoring:\n{}\n\n(from {})\n'
    .format(sup_prep_arrow_spark_adf__score, arrow_spark_adf))


# 38
sup_prep_arrow_spark_adf__score__vectorized = \
    sup_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='score',
        __vectorize__=True)
print('SupervisedBlueprint-prepped ArrowSparkADF for Scoring (Vectorized):\n{}\n\n(from {})\n'
    .format(sup_prep_arrow_spark_adf__score__vectorized, arrow_spark_adf))


# 39
sup_prep_arrow_spark_adf__eval = \
    sup_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='eval',
        __vectorize__=False)
print('SupervisedBlueprint-prepped ArrowSparkADF for Eval:\n{}\n\n(from {})\n'
    .format(sup_prep_arrow_spark_adf__eval, arrow_spark_adf))


# 40
sup_prep_arrow_spark_adf__eval__vectorized = \
    sup_bp.prep_data(
        df=arrow_spark_adf,
        __mode__='eval',
        __vectorize__=True)
print('SupervisedBlueprint-prepped ArrowSparkADF for Eval (Vectorized):\n{}\n\n(from {})\n'
    .format(sup_prep_arrow_spark_adf__eval__vectorized, arrow_spark_adf))
