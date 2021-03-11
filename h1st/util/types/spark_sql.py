from pyspark.sql.types import (
    # Spark Type      |   simpleString   |   Python Type          |
    # ----------------|------------------|------------------------|
    DataType,         #                  |                        |
                      #                  |                        |
    NullType,         #   null           |   None                 |
                      #                  |                        |
    AtomicType,       #                  |                        |
                      #                  |                        |
    BooleanType,      #   boolean        |   bool                 |
                      #                  |                        |
    StringType,       #   string         |   str, unicode         |
                      #                  |                        |
    BinaryType,       #   binary         |   bytearrary           |
                      #                  |                        |
    NumericType,      #                  |                        |
                      #                  |                        |
    IntegralType,     #                  |                        |
    ByteType,         #   tinyint        |   int, long            |
    ShortType,        #   smallint       |   int, long            |
    IntegerType,      #   int            |   int, long            |
    LongType,         #   bigint         |   int, long            |
                      #                  |                        |
    FractionalType,   #                  |                        |
    FloatType,        #   float          |   float                |
    DoubleType,       #   double         |   float                |
    DecimalType,      #   decimal(...)   |   Decimal              |
                      #                  |                        |
    DateType,         #   date           |   date, datetime       |
    TimestampType,    #   timestamp      |   datetime, time       |
                      #                  |                        |
    # Complex Types   #                  |                        |
    ArrayType,        #   array<...>     |   tuple, list, array   |
    MapType,          #   map<...>       |   dict                 |
    StructField,      #   ... : ...      |                        |
    StructType,       #   struct<...>    |   tuple, list, dict    |

    _atomic_types, _all_atomic_types, _all_complex_types,
    _type_mappings,
    _array_signed_int_typecode_ctype_mappings, _array_unsigned_int_typecode_ctype_mappings, _array_type_mappings,
    _acceptable_types
)


__null_type = NullType()
_NULL_TYPE = __null_type.simpleString()
assert _NULL_TYPE == __null_type.typeName()


__bool_type = BooleanType()
_BOOL_TYPE = __bool_type.simpleString()
assert _BOOL_TYPE == __bool_type.typeName()


__str_type = StringType()
_STR_TYPE = __str_type.simpleString()
assert _STR_TYPE == __str_type.typeName()


__binary_type = BinaryType()
_BINARY_TYPE = __binary_type.simpleString()
assert _BINARY_TYPE == __binary_type.typeName()


__byte_type = ByteType()
_TINYINT_TYPE = __byte_type.simpleString()

__short_type = ShortType()
_SMALLINT_TYPE = __short_type.simpleString()

__int_type = IntegerType()
_INT_TYPE = __int_type.simpleString()
assert _INT_TYPE == int.__name__
assert __int_type.typeName().startswith(_INT_TYPE)

__long_type = LongType()
_BIGINT_TYPE = __long_type.simpleString()
assert __long_type.typeName() == 'long'

_INT_TYPES = \
    [_TINYINT_TYPE, _SMALLINT_TYPE,
     _INT_TYPE, _BIGINT_TYPE]


__float_type = FloatType()
_FLOAT_TYPE = __float_type.simpleString()
assert _FLOAT_TYPE == __float_type.typeName()

__double_type = DoubleType()
_DOUBLE_TYPE = __double_type.simpleString()
assert _DOUBLE_TYPE == __double_type.typeName()

_FLOAT_TYPES = [_FLOAT_TYPE, _DOUBLE_TYPE]


_NUM_TYPES = _INT_TYPES + _FLOAT_TYPES


_POSSIBLE_CAT_TYPES = [_BOOL_TYPE, _STR_TYPE] + _NUM_TYPES
_POSSIBLE_FEATURE_TYPES = _POSSIBLE_CAT_TYPES + _NUM_TYPES


__date_type = DateType()
_DATE_TYPE = __date_type.simpleString()
assert _DATE_TYPE == __date_type.typeName()

__timestamp_type = TimestampType()
_TIMESTAMP_TYPE = __timestamp_type.simpleString()
assert _TIMESTAMP_TYPE == __timestamp_type.typeName()

_DATETIME_TYPES = [_DATE_TYPE, _TIMESTAMP_TYPE]


__decimal_10_0_type = DecimalType(precision=10, scale=0)
_DECIMAL_10_0_TYPE = __decimal_10_0_type.simpleString()

__decimal_38_18_type = DecimalType(precision=38, scale=18)
_DECIMAL_38_18_TYPE = __decimal_38_18_type.simpleString()

_DECIMAL_TYPE_PREFIX = '{}('.format(DecimalType.typeName())


_ARRAY_TYPE_PREFIX = '{}<'.format(ArrayType.typeName())
_MAP_TYPE_PREFIX = '{}<'.format(MapType.typeName())
_STRUCT_TYPE_PREFIX = '{}<'.format(StructType.typeName())

_VECTOR_TYPE = 'vector'
