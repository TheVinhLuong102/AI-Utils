from __future__ import print_function

import datetime
import itertools
import pandas

import sys
if sys.version_info.major == 3:
    from functools import reduce

import arimo.backend
from arimo.df.spark import SparkADF
import arimo.debug


arimo.debug.ON = True


N_IDS = 3
N_TIME_STEPS_PER_ID = 10
N_ROWS = N_IDS * N_TIME_STEPS_PER_ID

ID_RANGE = range(1, N_IDS + 1)
TIME_ORD_RANGE = range(1, N_TIME_STEPS_PER_ID + 1)

YEAR = datetime.date.today().year


ID_COL_NAME = 'i'
TIME_COL_NAME = 't'
CONTENT_COL_NAME = 'x'

DATED_TBL_ALIAS_a = 'dated_tbl_A'
DATED_TBL_ALIAS_ati = 'dated_tbl_ATI'
DATED_TBL_ALIAS_ait = 'dated_tbl_AIT'

TIMESTAMPED_TBL_ALIAS_a = 'timestamped_tbl_A'
TIMESTAMPED_TBL_ALIAS_ati = 'timestamped_tbl_ATI'
TIMESTAMPED_TBL_ALIAS_ait = 'timestamped_tbl_AIT'


dated_df = pandas.DataFrame(
    data={
        ID_COL_NAME:
            reduce(
                lambda x, y: x + y,
                [N_TIME_STEPS_PER_ID * ['i{}'.format(i)]
                 for i in ID_RANGE]),

        TIME_COL_NAME:
            [datetime.date(year=YEAR, month=m, day=d)
             for m, d in itertools.product(ID_RANGE, TIME_ORD_RANGE)],

        CONTENT_COL_NAME:
            range(11, N_ROWS + 11)
    }).sample(
        n=N_ROWS,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None)


dated_adf_A = SparkADF.create(data=dated_df)
assert (dated_adf_A.alias is None) \
   and (dated_adf_A.iCol is None) \
   and (dated_adf_A.tCol is None) \
   and (dated_adf_A.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

dated_adf_A.alias = DATED_TBL_ALIAS_a
assert (dated_adf_A.alias == DATED_TBL_ALIAS_a) \
   and (dated_adf_A.iCol is None) \
   and (dated_adf_A.tCol is None) \
   and (dated_adf_A.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

print('{}:'.format(dated_adf_A))
dated_adf_A.show(schema=True)

print('{}:'.format(DATED_TBL_ALIAS_a))
arimo.backend.spark.sql('SELECT * FROM {}'.format(DATED_TBL_ALIAS_a)).show()


dated_adf_ATI = SparkADF.create(data=dated_df)
assert (dated_adf_ATI.alias is None) \
   and (dated_adf_ATI.iCol is None) \
   and (dated_adf_ATI.tCol is None) \
   and (dated_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

dated_adf_ATI.alias = DATED_TBL_ALIAS_ati
assert (dated_adf_ATI.alias == DATED_TBL_ALIAS_ati) \
   and (dated_adf_ATI.iCol is None) \
   and (dated_adf_ATI.tCol is None) \
   and (dated_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

dated_adf_ATI.tCol = TIME_COL_NAME
assert (dated_adf_ATI.alias == DATED_TBL_ALIAS_ati) \
   and (dated_adf_ATI.iCol is None) \
   and (dated_adf_ATI.tCol == TIME_COL_NAME) \
   and (dated_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

print('{} w/ Date Components:'.format(dated_adf_ATI))
dated_adf_ATI.show(schema=True, __tAuxCols__=True)

print('{} w/ Date Components:'.format(DATED_TBL_ALIAS_ati))
arimo.backend.spark.sql('SELECT * FROM {}'.format(DATED_TBL_ALIAS_ati)).show()

dated_adf_ATI.iCol = ID_COL_NAME
assert (dated_adf_ATI.alias == DATED_TBL_ALIAS_ati) \
   and (dated_adf_ATI.iCol == ID_COL_NAME) \
   and (dated_adf_ATI.tCol == TIME_COL_NAME) \
   and (dated_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

print('Time-Series {}:'.format(dated_adf_ATI))
dated_adf_ATI.show(schema=True, __partitionID__=True)

print('Time-Series {}:'.format(DATED_TBL_ALIAS_ati))
arimo.backend.spark.sql('SELECT * FROM {}'.format(DATED_TBL_ALIAS_ati)).show()


dated_adf_AIT = \
    SparkADF.create(
        data=dated_df,
        alias=DATED_TBL_ALIAS_ait,
        iCol=ID_COL_NAME,
        tCol=TIME_COL_NAME,
        tChunkLen=3)
assert (dated_adf_AIT.alias == DATED_TBL_ALIAS_ait) \
   and (dated_adf_AIT.iCol == ID_COL_NAME) \
   and (dated_adf_AIT.tCol == TIME_COL_NAME) \
   and (dated_adf_AIT.tChunkLen == 3)

print('Time-Series {}:'.format(dated_adf_AIT))
dated_adf_AIT.show(schema=True, __partitionID__=True)

print('Time-Series {}:'.format(DATED_TBL_ALIAS_ait))
arimo.backend.spark.sql('SELECT * FROM {}'.format(DATED_TBL_ALIAS_ait)).show()


timestamped_df = pandas.DataFrame(
    data={
        ID_COL_NAME:
            reduce(
                lambda x, y: x + y,
                [N_TIME_STEPS_PER_ID * ['i{}'.format(i)]
                 for i in ID_RANGE]),

        TIME_COL_NAME:
            ['{}-{:02d}-{:02d}'.format(YEAR, m, d)
             for m, d in itertools.product(ID_RANGE, TIME_ORD_RANGE)],

        CONTENT_COL_NAME:
            range(11, N_ROWS + 11)
    }).sample(
        n=N_ROWS,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None)


timestamped_adf_A = SparkADF.create(data=timestamped_df)
assert (timestamped_adf_A.alias is None) \
   and (timestamped_adf_A.iCol is None) \
   and (timestamped_adf_A.tCol is None) \
   and (timestamped_adf_A.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

timestamped_adf_A.alias = TIMESTAMPED_TBL_ALIAS_a
assert (timestamped_adf_A.alias == TIMESTAMPED_TBL_ALIAS_a) \
   and (timestamped_adf_A.iCol is None) \
   and (timestamped_adf_A.tCol is None) \
   and (timestamped_adf_A.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

print('{}:'.format(timestamped_adf_A))
timestamped_adf_A.show(schema=True)

print('{}:'.format(TIMESTAMPED_TBL_ALIAS_a))
arimo.backend.spark.sql('SELECT * FROM {}'.format(TIMESTAMPED_TBL_ALIAS_a)).show()


timestamped_adf_ATI = SparkADF.create(data=timestamped_df)
assert (timestamped_adf_ATI.alias is None) \
   and (timestamped_adf_ATI.iCol is None) \
   and (timestamped_adf_ATI.tCol is None) \
   and (timestamped_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

timestamped_adf_ATI.alias = TIMESTAMPED_TBL_ALIAS_ati
assert (timestamped_adf_ATI.alias == TIMESTAMPED_TBL_ALIAS_ati) \
   and (timestamped_adf_ATI.iCol is None) \
   and (timestamped_adf_ATI.tCol is None) \
   and (timestamped_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

timestamped_adf_ATI.tCol = TIME_COL_NAME
assert (timestamped_adf_ATI.alias == TIMESTAMPED_TBL_ALIAS_ati) \
   and (timestamped_adf_ATI.iCol is None) \
   and (timestamped_adf_ATI.tCol == TIME_COL_NAME) \
   and (timestamped_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

print('{} w/ Date-Time Components:'.format(timestamped_adf_ATI))
timestamped_adf_ATI.show(schema=True, __tAuxCols__=True)

print('{} w/ Date-Time Components:'.format(TIMESTAMPED_TBL_ALIAS_ati))
arimo.backend.spark.sql('SELECT * FROM {}'.format(TIMESTAMPED_TBL_ALIAS_ati)).show()

timestamped_adf_ATI.iCol = ID_COL_NAME
assert (timestamped_adf_ATI.alias == TIMESTAMPED_TBL_ALIAS_ati) \
   and (timestamped_adf_ATI.iCol == ID_COL_NAME) \
   and (timestamped_adf_ATI.tCol == TIME_COL_NAME) \
   and (timestamped_adf_ATI.tChunkLen == SparkADF._DEFAULT_T_CHUNK_LEN)

print('Time-Series {}:'.format(timestamped_adf_ATI))
timestamped_adf_ATI.show(schema=True, __partitionID__=True)

print('Time-Series {}:'.format(TIMESTAMPED_TBL_ALIAS_ati))
arimo.backend.spark.sql('SELECT * FROM {}'.format(TIMESTAMPED_TBL_ALIAS_ati)).show()


timestamped_adf_AIT = \
    SparkADF.create(
        data=timestamped_df,
        alias=TIMESTAMPED_TBL_ALIAS_ait,
        iCol=ID_COL_NAME,
        tCol=TIME_COL_NAME,
        tChunkLen=3)
assert (timestamped_adf_AIT.alias == TIMESTAMPED_TBL_ALIAS_ait) \
   and (timestamped_adf_AIT.iCol == ID_COL_NAME) \
   and (timestamped_adf_AIT.tCol == TIME_COL_NAME) \
   and (timestamped_adf_AIT.tChunkLen == 3)

print('Time-Series {}:'.format(timestamped_adf_AIT))
timestamped_adf_AIT.show(schema=True, __partitionID__=True)

print('Time-Series {}:'.format(TIMESTAMPED_TBL_ALIAS_ait))
arimo.backend.spark.sql('SELECT * FROM {}'.format(TIMESTAMPED_TBL_ALIAS_ait)).show()
