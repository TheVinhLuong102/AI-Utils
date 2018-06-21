from __future__ import print_function

import datetime
import itertools
import pandas

from arimo.df.spark import SparkADF

import sys
if sys.version_info.major == 3:
    from functools import reduce


N_IDS = 3
N_TIME_STEPS_PER_ID = 10
N_ROWS = N_IDS * N_TIME_STEPS_PER_ID

ID_RANGE = range(1, N_IDS + 1)
TIME_ORD_RANGE = range(1, N_TIME_STEPS_PER_ID + 1)

YEAR = datetime.date.today().year


ID_COL_NAME = 'i'
TIME_COL_NAME = 't'
CONTENT_COL_NAME = 'x'


sparkDF = SparkADF.create(
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
    tCol=TIME_COL_NAME) \
    ._sparkDF


print('Full SparkDF is ORDERED')
sparkDF.show()

print('*** BUT subset SparkDF is UNORDERED!!! ***')
sparkDF[ID_COL_NAME, TIME_COL_NAME, CONTENT_COL_NAME].show()
