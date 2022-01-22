"""Pickle utilities."""


import pickle
from sys import version_info
from typing import Any


__all__ = (
    'COMPAT_PROTOCOL', 'COMPAT_COMPRESS',
    'DEFAULT_COMPRESS_LVL', 'MAX_COMPRESS_LVL',
    'PKL_EXT', 'PKL_W_PY_VER',
    'pickle_able',
)


COMPAT_PROTOCOL = 2
COMPAT_COMPRESS = 'bz2'   # smaller file size than ZLib/GZip

DEFAULT_COMPRESS_LVL = 3
MAX_COMPRESS_LVL = 9

PKL_EXT = '.pkl'
PKL_W_PY_VER = f'{PKL_EXT}{version_info.major}'


def pickle_able(obj: Any) -> bool:
    """Determine if object is picklable."""
    try:
        s = pickle.dumps(obj)   # pylint: disable=invalid-name
        _ = pickle.loads(s)
        return True

    except Exception as err:   # pylint: disable=broad-except
        print(err)
        return False
