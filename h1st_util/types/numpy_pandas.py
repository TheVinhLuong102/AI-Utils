"""NumPy & Pandas data types."""


import numpy

# pylint: disable=unused-import
from pandas.api.types import (   # noqa: F401
    is_extension_type,
    is_bool_dtype,
    is_categorical_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_datetime64_dtype,
    is_datetime64_ns_dtype,
    is_datetime64tz_dtype,
    is_extension_array_dtype,
    is_float_dtype,
    is_int64_dtype,
    is_integer_dtype,
    is_interval_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_period_dtype,
    is_signed_integer_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
    is_timedelta64_ns_dtype,
    is_unsigned_integer_dtype,
)


__all__ = (
    'NUMPY_FLOAT_TYPES', 'NUMPY_INT_TYPES', 'NUMPY_NUM_TYPES',
    'PY_NUM_TYPES',
    'FLOAT_TYPES', 'INT_TYPES', 'NUM_TYPES',
)


NUMPY_FLOAT_TYPES = (
    numpy.float_,
    numpy.float16,
    numpy.float32,
    numpy.float64,
    numpy.float128,
)

NUMPY_INT_TYPES = (
    numpy.int_,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.longlong,
    numpy.uint64,
    numpy.uint,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
    numpy.ulonglong,
)

NUMPY_NUM_TYPES = NUMPY_FLOAT_TYPES + NUMPY_INT_TYPES


PY_NUM_TYPES = float, int


FLOAT_TYPES = (float,) + NUMPY_FLOAT_TYPES
INT_TYPES = (int,) + NUMPY_INT_TYPES
NUM_TYPES = FLOAT_TYPES + INT_TYPES
