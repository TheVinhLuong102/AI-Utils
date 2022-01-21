from __future__ import print_function

import pickle
import sys


COMPAT_PROTOCOL = 2
COMPAT_COMPRESS = 'bz2'   # smaller file size than ZLib/GZip

DEFAULT_COMPRESS_LVL = 3
MAX_COMPRESS_LVL = 9

PKL_EXT = '.pkl'
PKL_W_PY_VER = PKL_EXT + str(sys.version_info.major)


def pickle_able(obj):
    try:
        s = pickle.dumps(obj)
        _ = pickle.loads(s)
        return True

    except Exception as err:
        print(err)
        return False
