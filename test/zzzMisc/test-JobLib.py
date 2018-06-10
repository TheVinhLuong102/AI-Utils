from __future__ import print_function

import joblib
import os
import pandas
import six

from arimo.util.pkl import COMPAT_PROTOCOL, COMPAT_COMPRESS, MAX_COMPRESS_LVL

if six.PY2:
    import cPickle as pickle

    PKL_VER = 'Pkl2'
    JOBLIB_VER = 'JobLib2'

else:
    import pickle

    PKL_VER = 'Pkl3'
    JOBLIB_VER = 'JobLib3'


TMP_DIR_PATH = '/tmp'

PKL_2_FILE_NAME = '2.pkl'
PKL_2_FILE_PATH = \
    os.path.join(
        TMP_DIR_PATH,
        PKL_2_FILE_NAME)

JOBLIB_2_FILE_NAME = '2.bz2'
JOBLIB_2_FILE_PATH = \
    os.path.join(
        TMP_DIR_PATH,
        JOBLIB_2_FILE_NAME)

PKL_3_FILE_NAME = '3.pkl'
PKL_3_FILE_PATH = \
    os.path.join(
        TMP_DIR_PATH,
        PKL_3_FILE_NAME)

JOBLIB_3_FILE_NAME = '3.bz2'
JOBLIB_3_FILE_PATH = \
    os.path.join(
        TMP_DIR_PATH,
        JOBLIB_3_FILE_NAME)


df = pandas.DataFrame(
    data=dict(
        x=[None, 0, 1]))

pickle.dump(
    df,
    open(PKL_2_FILE_PATH if six.PY2 else PKL_3_FILE_PATH, 'wb'),
    protocol=COMPAT_PROTOCOL)

joblib.dump(
    df,
    filename=JOBLIB_2_FILE_PATH if six.PY2 else JOBLIB_3_FILE_PATH,
    compress=(COMPAT_COMPRESS, MAX_COMPRESS_LVL),
    protocol=COMPAT_PROTOCOL)


if os.path.isfile(PKL_2_FILE_PATH):
    print('\nPkl2 opened by {}:'.format(PKL_VER),
          pickle.load(open(PKL_2_FILE_PATH, 'rb'))
          if six.PY2
          else pickle.load(
              open(PKL_2_FILE_PATH, 'rb'),
              encoding='latin1'),
          sep='\n')

    print('\nPkl2 opened by {}:'.format(JOBLIB_VER))
    if six.PY2:
        print(joblib.load(filename=PKL_2_FILE_PATH))

    else:
        try:
            joblib.load(filename=PKL_2_FILE_PATH)
        except Exception as err:
            print(err)


if os.path.isfile(JOBLIB_2_FILE_PATH):
    print('\nJobLib2 opened by {}:'.format(JOBLIB_VER),
          joblib.load(filename=JOBLIB_2_FILE_PATH),
          sep='\n')


if os.path.isfile(PKL_3_FILE_PATH):
    print('\nPkl3 opened by {}:'.format(PKL_VER),
          pickle.load(open(PKL_3_FILE_PATH, 'rb'))
          if six.PY2
          else pickle.load(
              open(PKL_3_FILE_PATH, 'rb'),
              encoding='utf-8'),
          sep='\n')

    print('\nPkl3 opened by {}:'.format(JOBLIB_VER),
          joblib.load(filename=PKL_3_FILE_PATH),
          sep='\n')


if os.path.isfile(JOBLIB_3_FILE_PATH):
    print('\nJobLib3 opened by {}:'.format(JOBLIB_VER),
          joblib.load(filename=JOBLIB_3_FILE_PATH),
          sep='\n')
