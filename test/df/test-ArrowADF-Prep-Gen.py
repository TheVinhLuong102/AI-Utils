import numpy
from pprint import pprint
import tqdm
import yaml

from arimo.IoT.DataAdmin import project


PROJECT = project('PanaAP-CC')


EQ_GEN_TP = 'refrig'   # 'disp_case'

if EQ_GEN_TP == 'disp_case':
    EQ_UNQ_TP_GRP = 'ex_display_case'
    LABEL_VAR = 'inside_temperature'
else:
    EQ_UNQ_TP_GRP = 'inverter_4_multi_comp_refrigerator'
    LABEL_VAR = 'suction_temperature'


SER_DIR_PATH = '/tmp/ADF.prep'


SEQ_LEN = 1   # 9

CHUNK_SIZE = 10 ** 5

BATCH_SIZE = 10 ** 3

N_BATCHES = 10 ** 3

N_THREADS = 4


# Load ArrowADF
arrow_adf = PROJECT.load_equipment_data(
    '{}---{}'.format(EQ_GEN_TP.upper(), EQ_UNQ_TP_GRP),
    set_i_col=SEQ_LEN > 1)

print(arrow_adf)


# Auto Profile & Prep ArrowADF
# and Serializing the Cat & Num Transform Pipelines to Files
prep_arrow_adf, cat_orig_to_prep_col_map, num_orig_to_prep_col_map = \
    arrow_adf.prep(
        scaleCat=False,
        savePath=SER_DIR_PATH,
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


# Deserialize & Apply Prep Transforms from Files
loaded_prep_arrow_adf, loaded_cat_orig_to_prep_col_map, loaded_num_orig_to_prep_col_map = \
    arrow_adf.prep(
        loadPath=SER_DIR_PATH,
        returnOrigToPrepColMaps=True,
        verbose=True)

print(loaded_prep_arrow_adf)


# ArrowADF.gen(...): Generating Batches for DL Training
gen_instance = \
    prep_arrow_adf.gen(
        cat_prep_cols + num_prep_cols +
        ((- SEQ_LEN + 1, 0)
         if SEQ_LEN > 1
         else ()),
        LABEL_VAR,
        n=BATCH_SIZE,
        sampleN=CHUNK_SIZE,
        withReplacement=False,
        seed=None,
        anon=True,
        collect='numpy',
        pad=numpy.nan,
        filter={LABEL_VAR: (-100, 100)},
        nThreads=N_THREADS)

print(gen_instance)


g0 = gen_instance()

x, y = next(g0)
pprint(gen_instance.colsLists)
print(x.shape, y.shape)

# Single-Process Multi-Threaded Throughput
for _ in tqdm.tqdm(range(N_BATCHES)):
    x, y = next(g0)


# ArrowADF.CrossSectDLDF(...) = instance of arimo.dl.reader.S3ParquetDatasetQueueReader
cross_sect_dldf = \
    prep_arrow_adf._CrossSectDLDF(
        feature_cols=cat_prep_cols + num_prep_cols,
        target_col=LABEL_VAR,
        n=BATCH_SIZE,
        sampleN=CHUNK_SIZE,
        filter={LABEL_VAR: (-100, 100)},
        nThreads=N_THREADS)

print(cross_sect_dldf)


g1 = cross_sect_dldf.generate_chunk()

chunk_df = next(g1)
pprint(chunk_df.columns.tolist())
print(chunk_df.shape)

# Single-Process Multi-Threaded Throughput
for _ in tqdm.tqdm(range(N_BATCHES * BATCH_SIZE // CHUNK_SIZE)):
    chunk_df = next(g1)
