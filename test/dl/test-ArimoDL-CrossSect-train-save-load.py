from __future__ import print_function

import yaml

from arimo.dl.base import DataFramePreprocessor, LossPlateauLrDecay, ModelServingPersistence
from arimo.dl.cross_sectional import FfnResnetRegressor

from arimo.IoT.DataAdmin import project


PROJECT = project('PanaAP-CC')


EQ_GEN_TP = 'disp_case'
EQ_UNQ_TP_GRP = 'n_micon_pcu_display_case'
LABEL_VAR = 'inside_temperature'


CHUNK_SIZE = 10 ** 5

BATCH_SIZE = 10 ** 3

N_BATCHES = 10 ** 3

N_THREADS = 4


PERSIST_DIR_PATH = '/tmp/.arimo.dl.model'


# Load ArrowADF
arrow_adf = PROJECT.load_equipment_data(
    '{}---{}'.format(EQ_GEN_TP.upper(), EQ_UNQ_TP_GRP),
    set_i_col=False)

print(arrow_adf)


# Auto Profile & Prep ArrowADF
# and Serializing the Cat & Num Transform Pipelines to Files
prep_arrow_adf, cat_orig_to_prep_col_map, num_orig_to_prep_col_map = \
    arrow_adf.prep(
        scaleCat=False,
        returnOrigToPrepColMaps=True,
        verbose=True)

print(prep_arrow_adf)


# show Profiled & Prep'ed Cat Cols
cat_cols, cat_prep_cols = \
    zip(*((cat_col, cat_orig_to_prep_col_map[cat_col][0])
          for cat_col in set(cat_orig_to_prep_col_map).difference(('__OHE__', '__SCALE__'))))

print('PROFILED CAT COLS: {}\n'.format(cat_cols))
print("PREP'ED CAT COLS: {}\n".format(cat_prep_cols))
print(yaml.safe_dump(cat_orig_to_prep_col_map))


# show Profiled & Prep'ed Num Cols
num_cols, num_prep_cols = \
    zip(*((num_col, num_orig_to_prep_col_map[num_col][0])
          for num_col in set(num_orig_to_prep_col_map).difference(('__SCALER__',))))

print('PROFILED NUM COLS: {}\n'.format(num_cols))
print("PREP'ED NUM COLS: {}\n".format(num_prep_cols))
print(yaml.safe_dump(num_orig_to_prep_col_map))


feature_cols = cat_prep_cols + num_prep_cols


# ArrowADF._CrossSectDLDF(...) = instance of arimo.dl.reader.S3ParquetDatasetQueueReader
cross_sect_dldf = \
    prep_arrow_adf._CrossSectDLDF(
        feature_cols=feature_cols,
        target_col=LABEL_VAR,
        n=BATCH_SIZE,
        sampleN=CHUNK_SIZE,
        filter={LABEL_VAR: (-100, 100)},
        nThreads=N_THREADS,
        isRegression=True)

print(cross_sect_dldf)


# Arimo.DL CrossSect Model
model = FfnResnetRegressor(
    hidden_size=1024,
    num_residual_blocks=4,
    residual_block_keep_prob=0.95,
    residual_layer_keep_prob=0.9,
    hidden_size_multiplier=0.5,
    learning_rate=1e-4,
    batch_size=512,
    vocab_size=None,
    embedding_size=None)

model.train_with_queue_reader_inputs(
    train_input=cross_sect_dldf,
    val_input=None,
    lr_scheduler=
        LossPlateauLrDecay(
            learning_rate=model.config.learning_rate,
            decay_rate=model.config.lr_decay,
            patience=1),
    max_epoch=1,
    early_stopping_patience=1,
    num_train_batches_per_epoch=N_BATCHES,
    num_test_batches_per_epoch=None)

print(model)


# Save & Load
ModelServingPersistence(
    model=model,
    preprocessor=
        DataFramePreprocessor(
            feature_cols=feature_cols,
            target_col=LABEL_VAR,
            num_targets=1,
            embedding_col=None,
            normalization=None)) \
.save(path=PERSIST_DIR_PATH)

print(ModelServingPersistence.load(path=PERSIST_DIR_PATH).model)
