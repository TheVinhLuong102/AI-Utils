from pyarrow.lib import \
    DataType, \
    NA, Type_NA, _NULL, null, \
    bool_, Type_BOOL, \
    string, Type_STRING, \
    binary, Type_BINARY, \
    FixedSizeBinaryType, Type_FIXED_SIZE_BINARY, \
    uint8, Type_UINT8, \
    int8, Type_INT8, \
    uint16, Type_UINT16, \
    int16, Type_INT16, \
    uint32, Type_UINT32, \
    int32, Type_INT32, \
    uint64, Type_UINT64, \
    int64, Type_INT64, \
    float16, Type_HALF_FLOAT, \
    float32, Type_FLOAT, \
    float64, Type_DOUBLE, \
    decimal128, Decimal128Type, Type_DECIMAL, \
    date32, Type_DATE32, \
    date64, Type_DATE64, \
    timestamp, TimestampType, Type_TIMESTAMP, \
    time32, Time32Type, Type_TIME32, \
    time64, Time64Type, Type_TIME64, \
    list_, ListType, Type_LIST, \
    struct, StructType, Type_STRUCT, \
    union, UnionType, Type_UNION, \
    dictionary, DictionaryType, Type_DICTIONARY, \
    Type_MAP, \
    is_boolean_value, is_float_value, is_integer_value, is_named_tuple

from pyarrow.types import \
    is_null, \
    is_boolean, \
    is_string, is_unicode, \
    is_binary, is_fixed_size_binary, \
    _SIGNED_INTEGER_TYPES, _UNSIGNED_INTEGER_TYPES, _INTEGER_TYPES, \
    is_signed_integer, is_unsigned_integer, is_integer, \
    is_int8, is_int16, is_int32, is_int64, is_uint8, is_uint16, is_uint32, is_uint64, \
    _FLOATING_TYPES, is_floating, is_float16, is_float32, is_float64, \
    is_decimal, \
    _DATE_TYPES, is_date, is_date32, is_date64, \
    _TIME_TYPES, is_time, is_time32, is_time64, \
    is_timestamp, \
    _TEMPORAL_TYPES, is_temporal, \
    _NESTED_TYPES, is_list, is_struct, is_union, is_map, is_nested, \
    is_dictionary

from .spark_sql import \
    _NULL_TYPE, \
    _BOOL_TYPE, \
    _STR_TYPE, \
    _BINARY_TYPE, \
    _INT_TYPE, _BIGINT_TYPE, \
    _FLOAT_TYPE, _DOUBLE_TYPE, \
    _DATE_TYPE, _TIMESTAMP_TYPE, \
    _VECTOR_TYPE, \
    _DECIMAL_TYPE_PREFIX, \
    _ARRAY_TYPE_PREFIX, _MAP_TYPE_PREFIX, _STRUCT_TYPE_PREFIX


_ARROW_NULL_TYPE = null()
assert str(_ARROW_NULL_TYPE) == _NULL_TYPE


_ARROW_BOOL_TYPE = bool_()
__arrow_bool_type_str = str(_ARROW_BOOL_TYPE)
assert __arrow_bool_type_str == bool.__name__
assert _BOOL_TYPE.startswith(__arrow_bool_type_str)


_ARROW_STR_TYPE = string()
assert str(_ARROW_STR_TYPE) == _STR_TYPE


_ARROW_BINARY_TYPE = binary(-1)
assert str(_ARROW_BINARY_TYPE) == _BINARY_TYPE


_ARROW_INT_TYPE = int64()
_ARROW_DOUBLE_TYPE = float64()

_ARROW_DATE_TYPE = date32()
_ARROW_TIMESTAMP_TYPE = timestamp(unit='ns', tz=None)


def is_float(arrow_type):
    return is_floating(arrow_type) or is_decimal(arrow_type)


def is_num(arrow_type):
    return is_integer(arrow_type) or is_float(arrow_type)


def is_possible_cat(arrow_type):
    return is_boolean(arrow_type) or is_string(arrow_type) or is_num(arrow_type)


def is_complex(arrow_type):
    return is_dictionary(arrow_type) or is_nested(arrow_type)
