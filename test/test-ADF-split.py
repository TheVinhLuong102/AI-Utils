from __future__ import print_function

import datetime
import itertools
import pandas

import sys
if sys.version_info.major == 3:
    from functools import reduce

from arimo.df.spark import ADF


N_IDS = 3
N_TIME_STEPS_PER_ID = 20
N_ROWS = N_IDS * N_TIME_STEPS_PER_ID

ID_RANGE = range(1, N_IDS + 1)
TIME_ORD_RANGE = range(1, N_TIME_STEPS_PER_ID + 1)

YEAR = datetime.date.today().year

ID_COL_NAME = 'i'
TIME_COL_NAME = 't'
CONTENT_COL_NAME = 'x'

SPLIT_ADF_T_CHUNK_LEN = 3
ORIG_T_CHUNK_LEN = 2 * SPLIT_ADF_T_CHUNK_LEN


adf = ADF.create(
    data=pandas.DataFrame(
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
            axis=None),
    iCol=ID_COL_NAME,
    tCol=TIME_COL_NAME,
    tChunkLen=ORIG_T_CHUNK_LEN)


adf_1, adf_2 = adf.split(.5, .5)

adf_1.tChunkLen = adf_2.tChunkLen = SPLIT_ADF_T_CHUNK_LEN

print('Split ADF #1:')
adf_1.show()

print('Split ADF #2')
adf_2.show()
