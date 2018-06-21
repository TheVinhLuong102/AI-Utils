from __future__ import print_function

import numpy
import pandas
import random

import arimo.backend
from arimo.df.spark import SparkADF
from arimo.util import fs


PERSIST_BASEPATH = '/tmp/test-adf'


N_IN_MEMORY_PARTITIONS = random.choice([200, 2001])


Ns_FILE_PARTITIONS = 1, 100, 300, 1000, 3000


_UNCOMPRESSED_COMPRESSION_CODECS = \
    (# None,   # java.lang.NullPointerException
     'none',
     'uncompressed')


FORMATS_N_COMPRESSION_CODECS = \
    ('parquet', _UNCOMPRESSED_COMPRESSION_CODECS +
                ('gzip',
                 # 'lzo',   # com.hadoop.compression.lzo.LzoCodec not found
                 'snappy')), \
    ('orc', _UNCOMPRESSED_COMPRESSION_CODECS +
            (# 'lzo',   # com.hadoop.compression.lzo.LzoCodec not found
             'snappy',
             'zlib')), \
    ('tfrecords', _UNCOMPRESSED_COMPRESSION_CODECS), \
    ('avro', _UNCOMPRESSED_COMPRESSION_CODECS + ('deflate', 'snappy')), \
    ('csv', _UNCOMPRESSED_COMPRESSION_CODECS +
            ('bzip2',
             'deflate',
             'gzip',
             'lz4',
             # 'snappy'   # SnappyCompressor has not been loaded
            )), \
    ('json', _UNCOMPRESSED_COMPRESSION_CODECS +
            ('bzip2',
             'deflate',
             'gzip',
             'lz4',
             # 'snappy'   # SnappyCompressor has not been loaded
            ))
    # ('text', _UNCOMPRESSED_COMPRESSION_CODECS + ('bzip2', 'deflate', 'gzip', 'lz4', 'snappy'))


FULL_FORMAT_NAMES = dict(avro='com.databricks.spark.avro')


arimo.backend.init(
    sparkApp='test SparkADF save/load ({} in-memory partitions)'.format(N_IN_MEMORY_PARTITIONS),
    sparkConf={'spark.default.parallelism': N_IN_MEMORY_PARTITIONS})


TEST_ADF = SparkADF.create(
    data=pandas.DataFrame(
        data=dict(
            boolX=[None, False, True],
            fltX=[None, numpy.nan, 10.],
            intX=[-1, 0, 1],
            strX=[None, 'a', 'b'],
            boolA=[None,
                   [],
                   [False, True]],
            fltA=[None,
                  [],
                  [numpy.nan, 0., 10.]],
            intA=[None,
                  [],
                  [0, 1]],
            strA=[None,
                  [],
                  ['a', 'b']]
        )))(
    'boolX',
    'IF(boolX IS NULL, NULL, fltX) AS fltX',
    'IF(boolX IS NULL, NULL, intX) AS intX',
    'strX',
    'boolA', 'fltA', 'intA', 'strA',
    'ARRAY(boolA, boolA) AS boolAA',
    'ARRAY(fltA, fltA) AS fltAA',
    'ARRAY(intA, intA) AS intAA',
    'ARRAY(strA, strA) AS strAA',
    '_ARRAY_TO_VECTOR(boolA) AS boolV',
    '_ARRAY_TO_VECTOR(fltA) AS fltV',
    '_ARRAY_TO_VECTOR(intA) AS intV')(
    '*',
    'ARRAY(boolV, fltV, intV) AS vA')

assert TEST_ADF.rdd.getNumPartitions() == N_IN_MEMORY_PARTITIONS

print(TEST_ADF)
TEST_ADF.printSchema()
TEST_ADF.cache()
TEST_ADF.show()


ADFS_FOR_FORMATS = \
    dict(parquet=TEST_ADF,
         orc=TEST_ADF,

         tfrecords=TEST_ADF.filter('boolX IS NOT NULL').rm(
             'boolX',   # Cannot convert field to unsupported data type BooleanType
             'boolA',   # Cannot convert field to unsupported data type ArrayType(BooleanType(...,true),true)
             'boolAA',   # Cannot convert field to unsupported data type ArrayType(ArrayType(...,true),true)
             'fltAA',   # Cannot convert field to unsupported data type ArrayType(ArrayType(...,true),true)
             'intAA',   # Cannot convert field to unsupported data type ArrayType(ArrayType(...,true),true)
             'strAA',   # Cannot convert field to unsupported data type ArrayType(ArrayType(...,true),true)
             'vA'   # Cannot convert field to unsupported data type ArrayType(org.apache.spark.ml.linalg.VectorUDT,true)
         ),

         avro=TEST_ADF.rm(
             'boolV',   # Unexpected type org.apache.spark.ml.linalg.VectorUDT
             'fltV',   # Unexpected type org.apache.spark.ml.linalg.VectorUDT
             'intV',   # Unexpected type org.apache.spark.ml.linalg.VectorUDT
             'vA'   # Unexpected type org.apache.spark.ml.linalg.VectorUDT
         ),

         csv=TEST_ADF.rm(
            'boolA',   # CSV data source does not support array<...> data type
            'fltA',   # CSV data source does not support array<...> data type
            'intA',   # CSV data source does not support array<...> data type
            'strA',   # CSV data source does not support array<...> data type
            'boolAA',   # CSV data source does not support array<...> data type
            'fltAA',   # CSV data source does not support array<...> data type
            'intAA',   # CSV data source does not support array<...> data type
            'strAA',   # CSV data source does not support array<...> data type
            'boolV',   # CSV data source does not support struct<...> data type
            'fltV',   # CSV data source does not support struct<...> data type
            'intV',   # CSV data source does not support struct<...> data type
            'vA'   # CSV data source does not support array<...> data type
         ),

         json=TEST_ADF)


for fmt, compression_codecs in FORMATS_N_COMPRESSION_CODECS:
    if fmt in ADFS_FOR_FORMATS:
        for compression_codec in compression_codecs:
            adf = ADFS_FOR_FORMATS[fmt]
            full_format_name = FULL_FORMAT_NAMES.get(fmt, fmt)
            for n_file_partitions in Ns_FILE_PARTITIONS:
                path = '{}.{}-partition.{}.{}'.format(PERSIST_BASEPATH, n_file_partitions, fmt, compression_codec)
                if fmt == 'tfrecords':
                    fs.rm(path=path, is_dir=True, hdfs=fs._ON_LINUX_CLUSTER_WITH_HDFS)
                adf.repartition(n_file_partitions) \
                    .save(path=path, mode='overwrite',
                          format=full_format_name, compression=compression_codec,
                          verbose=True)
                adf = SparkADF.load(path=path,
                               format=full_format_name,
                               schema=None,
                               verbose=True)
                n_in_memory_partitions = adf.rdd.getNumPartitions()
                assert n_in_memory_partitions \
                    == (min(n_file_partitions, TEST_ADF.nRows)
                        if (fmt == 'orc') or
                           ((fmt in ('csv', 'json')) and
                            (compression_codec in _UNCOMPRESSED_COMPRESSION_CODECS))
                        else n_file_partitions), \
                    '{}: {} IN-MEMORY PARTITIONS'.format(path, n_in_memory_partitions)
                print(adf)
                adf.show()
