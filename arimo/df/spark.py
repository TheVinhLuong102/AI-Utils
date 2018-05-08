"""
.. toctree::
"""
from __future__ import division, print_function

from argparse import Namespace as _Namespace
from collections import Counter
import copy
import itertools
import json
import numpy
import os
import pandas
import random
import tempfile
import time
import types
import uuid

import six
if six.PY2:
    from functools32 import lru_cache
    _NUM_CLASSES = int, long, float
    _STR_CLASSES = str, unicode
else:
    from functools import lru_cache
    _NUM_CLASSES = int, float
    _STR_CLASSES = str

import arimo.backend

from pyspark.ml import PipelineModel, Transformer

from pyspark.ml.feature import (
    Binarizer,
        # thresholding numerical features to binary (0/1) features
    Bucketizer,
        # transforms a column of continuous features to a column of feature buckets
    QuantileDiscretizer,
        # takes column with continuous features and outputs a column with binned categorical features
    StringIndexer,
        # encodes a string column of labels to a column of label indices
    IndexToString,
        # maps a column of label indices back to a column containing the original labels as strings
    OneHotEncoderEstimator as OneHotEncoder, OneHotEncoderModel,
        # maps a column of category indices to a column of binary vectors, with at most a single one-value
    VectorIndexer,
        # index categorical features in datasets of Vectors, both
        # automatically decide which features are categorical & convert original values to category indices

    Imputer,
        # Imputation estimator for completing missing values, either using the mean or the median of the columns in which the missing values are located.
    MaxAbsScaler,
        # transforms a dataset of Vector rows, rescaling each feature to range [-1, 1]
    MinMaxScaler,
        # transforms a dataset of Vector rows, rescaling each feature to a specific range (often [0, 1]).
    Normalizer,
        # transforms a dataset of Vector rows, normalizing each Vector to have unit norm
    StandardScaler,
        # transforms dataset of Vector rows, normalizing each feature to have unit stdev and/or zero mean

    VectorAssembler,
        # combines a given list of columns into a single vector column

    SQLTransformer)

from pyspark.ml.linalg import Vector
from pyspark.sql import DataFrame, functions as sparkSQLFuncs
from pyspark.sql.window import Window

from arimo.util import DefaultDict, fs, Namespace
from arimo.util.aws import rds, s3
from arimo.util.decor import enable_inplace, _docstr_settable_property, _docstr_verbose
from arimo.util.iterables import flatten, to_iterable
from arimo.util.types.spark_sql import \
    _INT_TYPE, _BIGINT_TYPE, _INT_TYPES, _DOUBLE_TYPE, _FLOAT_TYPES, _NUM_TYPES, \
    _BOOL_TYPE, _STR_TYPE, _POSSIBLE_CAT_TYPES, _DATE_TYPE, _TIMESTAMP_TYPE, \
    _VECTOR_TYPE, _DECIMAL_TYPE_PREFIX, _ARRAY_TYPE_PREFIX, _MAP_TYPE_PREFIX, _STRUCT_TYPE_PREFIX, \
    StructField, StructType
import arimo.debug

from . import _ADFABC


# decorator to add standard SparkADF keyword-arguments docstring
def _docstr_adf_kwargs(method):
    def f():
        """
        Keyword Args:
            alias (str, default = None): name of the ``SparkADF`` to register as a table in its ``SparkSession``

            iCol (str, default = 'id'): name of the ``SparkADF``'s *identity*/*entity* column, if applicable

            tCol (str, default = None): name of the ``SparkADF``'s *timestamp* column, if applicable

            tChunkLen (int, default = None): length of time-series chunks

            reprSampleSize (int, default = 1,000,000): *approximate* number of rows to sample from the ``SparkADF``
                for profiling purposes

            minNonNullProportion (float between 0 and 1, default = .32): minimum proportion of non-``NULL`` values
                in each column to qualify it as a valid feature to use in downstream data analyses

            outlierTailProportion (float between 0 and .1, default = .005): proportion in each tail end
                of each numerical column's distribution to exclude when computing outlier-resistant statistics

            maxNCats (int, default = 12): maximum number of categorical levels to consider
                for each possible categorical column

            minProportionByMaxNCats (float between 0 and 1, default = .9): minimum total proportion accounted for by the
                most common ``maxNCats`` of each possible categorical column to consider the column truly categorical
        """
    method.__doc__ += f.__doc__
    return method


@enable_inplace
class SparkADF(_ADFABC):
    """
    NOTE: Using `SparkADF` requires a cluster with Apache Spark set up.

    ``SparkADF`` extends the ``Spark SQL DataFrame``
    while still offering full access to ``Spark DataFrame`` functionalities.

    **IMPORTANT**: Any ``Spark DataFrame`` method not explicitly re-implemented in ``SparkADF``
    is a valid ``SparkADF`` method and works the same way as it does under ``Spark DataFrame``.
    If the method's result is a ``Spark DataFrame``, it is up-converted to an ``SparkADF``.
    """
    # Partition ID col
    _PARTITION_ID_COL = '__PARTITION_ID__'

    # extra aux cols
    _T_CHUNK_COL = '__tChunk__'
    _T_ORD_IN_CHUNK_COL = '__tOrd_inChunk__'

    _T_REL_AUX_COLS = _ADFABC._T_ORD_COL, _T_CHUNK_COL, _T_ORD_IN_CHUNK_COL, _ADFABC._T_DELTA_COL
    _T_AUX_COLS = _T_REL_AUX_COLS + _ADFABC._T_COMPONENT_AUX_COLS

    # default ordered chunk size for time-series SparkADFs
    _DEFAULT_T_CHUNK_LEN = 1000

    # default arguments dict
    _DEFAULT_KWARGS = \
        dict(
            alias=None,
            detPrePartitioned=False, nDetPrePartitions=None,
            iCol=_ADFABC._DEFAULT_I_COL, tCol=None,
            tChunkLen=_DEFAULT_T_CHUNK_LEN,
            reprSampleSize=_ADFABC._DEFAULT_REPR_SAMPLE_SIZE,
            minNonNullProportion=DefaultDict(_ADFABC._DEFAULT_MIN_NON_NULL_PROPORTION),
            outlierTailProportion=DefaultDict(_ADFABC._DEFAULT_OUTLIER_TAIL_PROPORTION),
            maxNCats=DefaultDict(_ADFABC._DEFAULT_MAX_N_CATS),
            minProportionByMaxNCats=DefaultDict(_ADFABC._DEFAULT_MIN_PROPORTION_BY_MAX_N_CATS))

    # repr sample
    _REPR_SAMPLE_ALIAS_SUFFIX = '__ReprSample'

    # "inplace-able" methods
    _INPLACE_ABLE = \
        '__call__',\
        '__getattr__', \
        'fillna', \
        'prep', \
        'rename', \
        'sample', \
        'select', \
        'sql'

    # whether to test loading Parquet on HDFS
    _TEST_HDFS_LOAD = not fs._ON_LINUX_CLUSTER_WITH_HDFS

    # ********************************
    # "INTERNAL / DON'T TOUCH" METHODS
    # __init__
    # _extractStdKwArgs
    # _organizeTimeSeries
    # _emptyCache
    # _inheritCache
    # _decorate
    # _inplace

    @_docstr_adf_kwargs
    def __init__(self, sparkDF, nRows=None, **kwargs):
        """
        Return:
            ``SparkADF`` instance

        Args:
            sparkDF: a ``Spark DataFrame``
        """
        # set Spark Session
        self._sparkSession = arimo.backend.spark

        # set underlying Spark SQL DataFrame
        self._sparkDF = sparkDF

        # initiate empty cache
        self._cache = _Namespace(
            nPartitions=sparkDF.rdd.getNumPartitions(),
            nRows=nRows)

        # extract standard keyword arguments
        self._extractStdKwArgs(kwargs, resetToClassDefaults=True, inplace=True)

        # organize time series if applicable
        self._organizeTimeSeries(forceGenTRelAuxCols=False)

        # register alias in SparkSession
        if self._alias:
            self.createOrReplaceTempView(name=self._alias)

        # set profiling settings and create empty profiling cache
        self._emptyCache()

    def _extractStdKwArgs(self, kwargs, resetToClassDefaults=False, inplace=False):
        nameSpace = self \
            if inplace \
            else _Namespace()

        for k, classDefaultV in self._DEFAULT_KWARGS.items():
            _privateK = '_{}'.format(k)

            if not resetToClassDefaults:
                existingInstanceV = getattr(self, _privateK, None)

            setattr(
                nameSpace,
                _privateK   # *** USE _k TO NOT INVOKE @k.setter RIGHT AWAY ***
                    if inplace
                    else k,
                kwargs.pop(
                    k,
                    existingInstanceV
                        if (not resetToClassDefaults) and existingInstanceV
                        else classDefaultV))

        if inplace:
            cols = self.columns

            if self._detPrePartitioned:
                if self._PARTITION_ID_COL in cols:
                    assert self._nDetPrePartitions

                else:
                    if self._nDetPrePartitions is None:
                        self._nDetPrePartitions = self.nPartitions
                    else:
                        assert self._nDetPrePartitions == self.nPartitions, \
                            '*** Deterministically Pre-Partitioned SparkADF: Set nDetPrePartitions {} != Detected {} Partitions (Likely Due to Big Files Being Split by Spark) ***'.format(
                                self._nDetPrePartitions, self.nPartitions)

                    self._sparkDF = \
                        self._sparkDF.withColumn(
                            colName=self._PARTITION_ID_COL,
                            col=sparkSQLFuncs.spark_partition_id())

            else:
                assert self._nDetPrePartitions is None

            if self._iCol not in cols:
                self._iCol = None

            if self._tCol not in cols:
                self._tCol = None
                
        else:
            return nameSpace

    def _organizeTimeSeries(self, forceGenTRelAuxCols=False):
        if self._tCol:
            tCol = self._tCol
            assert tCol in self.columns

            _typesCached = hasattr(self._cache, 'type')
            _types = {}
            _types[tCol] = _tColType = self.type(tCol)
            _tIsDate = False
            _tColExpr = self[tCol]
            _castTColType = False

            _tComponentColsApplicable = True

            if _tColType == _DATE_TYPE:
                self._dCol = dCol = tCol
                _dColAbsent = _genDCol = False
                _dColExpr = None
                _types[dCol] = _DATE_TYPE

                _tIsDate = True
                _tColExprForDelta = _tColExpr
                _tDeltaColType = _INT_TYPE

            elif _tColType == _TIMESTAMP_TYPE:
                self._dCol = dCol = self._DEFAULT_D_COL
                if dCol in self.columns:
                    _dColAbsent = False
                    _dColType = self.type(dCol)
                    if _dColType == _DATE_TYPE:
                        _genDCol = False
                    else:
                        assert _dColType == _STR_TYPE
                        _genDCol = True
                        _dColExpr = self[dCol].cast(dataType=_DATE_TYPE).alias(dCol)
                else:
                    _dColAbsent = _genDCol = True
                    _dColExpr = _tColExpr.cast(dataType=_DATE_TYPE).alias(dCol)
                _types[dCol] = _DATE_TYPE

                _tColExprForDelta = _tColExpr.cast(dataType=_DOUBLE_TYPE)
                _tDeltaColType = _DOUBLE_TYPE

            elif _tColType == _STR_TYPE:
                self._dCol = dCol = self._DEFAULT_D_COL
                if dCol in self.columns:
                    _dColAbsent = False
                    _dColType = self.type(dCol)
                    if _dColType == _DATE_TYPE:
                        _genDCol = False
                    else:
                        assert _dColType == _STR_TYPE
                        _genDCol = True
                        _dColExpr = self[dCol].cast(dataType=_DATE_TYPE).alias(dCol)
                else:
                    _dColAbsent = _genDCol = True
                    _dColExpr = _tColExpr.cast(dataType=_DATE_TYPE).alias(dCol)
                _types[dCol] = _DATE_TYPE

                _castTColType = True
                _types[tCol] = _TIMESTAMP_TYPE
                _tColExpr = _tColExpr.cast(dataType=_TIMESTAMP_TYPE).alias(tCol)
                _tColExprForDelta = _tColExpr.cast(dataType=_DOUBLE_TYPE)
                _tDeltaColType = _DOUBLE_TYPE

            elif _tColType in (_BIGINT_TYPE, _FLOAT_TYPES, _DOUBLE_TYPE):
                _tColExprForDelta = _tColExpr
                _tDeltaColType = _tColType
                _castTColType = True
                _types[tCol] = _TIMESTAMP_TYPE
                _tColExpr = _tColExpr.cast(dataType=_TIMESTAMP_TYPE).alias(tCol)

                _dColAbsent = False
                self._dCol = dCol = self._DEFAULT_D_COL
                if dCol in self.columns:
                    _dColType = self.type(dCol)
                    if _dColType == _DATE_TYPE:
                        _genDCol = False
                    else:
                        assert _dColType == _STR_TYPE
                        _genDCol = True
                        _dColExpr = self[dCol].cast(dataType=_DATE_TYPE).alias(dCol)
                else:
                    _genDCol = True
                    _dColExpr = _tColExpr.cast(dataType=_DATE_TYPE).alias(dCol)
                _types[dCol] = _DATE_TYPE

            elif _tColType in _INT_TYPES:   # if non-big Int
                _tColExprForDelta = _tColExpr
                _tDeltaColType = _tColType
                _tComponentColsApplicable = False

                _dColAbsent = False
                if self._DEFAULT_D_COL in self.columns:
                    self._dCol = dCol = self._DEFAULT_D_COL
                    _dColType = self.type(dCol)
                    if _dColType == _DATE_TYPE:
                        _genDCol = False
                    else:
                        assert _dColType == _STR_TYPE
                        _genDCol = True
                        _dColExpr = self[dCol].cast(dataType=_DATE_TYPE).alias(dCol)
                else:
                    self._dCol = dCol = _dColExpr = None
                    _genDCol = False

            else:
                raise TypeError(
                    '*** {}: Type of Time Column must be either Date, Timestamp, String, BigInt, Double, Float or Int ***'
                        .format(self))

            if dCol:
                _dColTup = dCol,

                if _genDCol:
                    self._sparkDF = \
                        self._sparkDF.select(
                            _dColExpr,
                            *(col for col in self.columns
                                  if col != dCol))
            else:
                _dColTup = ()

            if _tComponentColsApplicable:
                _tHoYColAbsent = \
                    self._T_HoY_COL not in self.columns
                _genTHoYCol = \
                    _tHoYColAbsent or \
                    (self.type(self._T_HoY_COL) not in _INT_TYPES)

                _tQoYColAbsent = \
                    self._T_QoY_COL not in self.columns
                _genTQoYCol = \
                    _tQoYColAbsent or \
                    (self.type(self._T_QoY_COL) not in _INT_TYPES)

                _tMoYColAbsent = \
                    self._T_MoY_COL not in self.columns
                _genTMoYCol = \
                    _tMoYColAbsent or \
                    (self.type(self._T_MoY_COL) not in _INT_TYPES)

                # _tWoYColAbsent = \
                #     self._T_WoY_COL not in self.columns
                # _genTWoYCol = \
                #     _tWoYColAbsent or \
                #     (self.type(self._T_WoY_COL) not in _INT_TYPES)

                # _tDoYColAbsent = \
                #     self._T_DoY_COL not in self.columns
                # _genTDoYCol = \
                #     _tDoYColAbsent or \
                #     (self.type(self._T_DoY_COL) not in _INT_TYPES)

                _tPoYColAbsent = \
                    self._T_PoY_COL not in self.columns
                _genTPoYCol = \
                    _tPoYColAbsent or \
                    (self.type(self._T_PoY_COL) not in _FLOAT_TYPES)

                _tQoHColAbsent = \
                    self._T_QoH_COL not in self.columns
                _genTQoHCol = \
                    _tQoHColAbsent or \
                    (self.type(self._T_QoH_COL) not in _INT_TYPES)

                _tMoHColAbsent = \
                    self._T_MoH_COL not in self.columns
                _genTMoHCol = \
                    _tMoHColAbsent or \
                    (self.type(self._T_MoH_COL) not in _INT_TYPES)

                _tPoHColAbsent = \
                    self._T_PoH_COL not in self.columns
                _genTPoHCol = \
                    _tPoHColAbsent or \
                    (self.type(self._T_PoH_COL) not in _FLOAT_TYPES)

                _tMoQColAbsent = \
                    self._T_MoQ_COL not in self.columns
                _genTMoQCol = \
                    _tMoQColAbsent or \
                    (self.type(self._T_MoQ_COL) not in _INT_TYPES)

                _tPoQColAbsent = \
                    self._T_PoQ_COL not in self.columns
                _genTPoQCol = \
                    _tPoQColAbsent or \
                    (self.type(self._T_PoQ_COL) not in _FLOAT_TYPES)

                _tWoMColAbsent = \
                    self._T_WoM_COL not in self.columns
                _genTWoMCol = \
                    _tWoMColAbsent or \
                    (self.type(self._T_WoM_COL) not in _INT_TYPES)

                _tDoMColAbsent = \
                    self._T_DoM_COL not in self.columns
                _genTDoMCol = \
                    _tDoMColAbsent or \
                    (self.type(self._T_DoM_COL) not in _INT_TYPES)

                _tPoMColAbsent = \
                    self._T_PoM_COL not in self.columns
                _genTPoMCol = \
                    _tPoMColAbsent or \
                    (self.type(self._T_PoM_COL) not in _FLOAT_TYPES)

                _tDoWColAbsent = \
                    self._T_DoW_COL not in self.columns
                _genTDoWCol = \
                    _tDoWColAbsent or \
                    (self.type(self._T_DoW_COL) != _STR_TYPE)

                _tPoWColAbsent = \
                    self._T_PoW_COL not in self.columns
                _genTPoWCol = \
                    _tPoWColAbsent or \
                    (self.type(self._T_PoW_COL) not in _FLOAT_TYPES)

                _tDailyComponentColsApplicable = not _tIsDate

                if _tDailyComponentColsApplicable:
                    _tHoDColAbsent = \
                        self._T_HoD_COL not in self.columns
                    _genTHoDCol = \
                        _tHoDColAbsent or \
                        (self.type(self._T_HoD_COL) not in _INT_TYPES)

                    _tPoDColAbsent = \
                        self._T_PoD_COL not in self.columns
                    _genTPoDCol = \
                        _tPoDColAbsent or \
                        (self.type(self._T_PoD_COL) not in _FLOAT_TYPES)

            contentCols = None

            if self._iCol:
                self.hasTS = True
                iCol = self._iCol
                assert iCol in self.columns

                _tOrdColAbsent = self._T_ORD_COL not in self.columns
                _tDeltaColAbsent = self._T_DELTA_COL not in self.columns

                if self._detPrePartitioned:
                    _tChunkColAbsent = _genTChunkCol = \
                        _tOrdInChunkColAbsent = _genTOrdInChunkCol = False

                    if _tOrdColAbsent:
                        _genTOrdCol = _genTDeltaCol = True

                        window = Window \
                            .partitionBy(iCol) \
                            .orderBy(tCol)

                    else:
                        _genTOrdCol = \
                            forceGenTRelAuxCols or \
                            (self.type(self._T_ORD_COL) not in _INT_TYPES)

                        _genTDeltaCol = \
                            _genTOrdCol or \
                            _tDeltaColAbsent or \
                            (self.type(self._T_DELTA_COL) != _tDeltaColType)

                        if _genTDeltaCol:
                            window = Window \
                                .partitionBy(iCol) \
                                .orderBy(tCol if _genTOrdCol else self._T_ORD_COL)

                    if arimo.debug.ON:
                        if forceGenTRelAuxCols:
                            self.class_stdout_logger().debug(
                                msg='*** FORCE-GENERATING AUXILIARY COLUMNS {}, {} ***'
                                    .format(self._T_ORD_COL, self._T_DELTA_COL))

                        elif not (_genTOrdCol and _genTDeltaCol):
                            self.class_stdout_logger().debug(
                                msg='*** SKIP GENERATING ALREADY EXISTING AUXILIARY COLUMN(S) {} ***'
                                    .format((() if _genTOrdCol
                                                else (self._T_ORD_COL,)) +
                                            (() if _genTDeltaCol
                                                else (self._T_DELTA_COL,))))

                else:
                    _tChunkColAbsent = self._T_CHUNK_COL not in self.columns
                    _tOrdInChunkColAbsent = self._T_ORD_IN_CHUNK_COL not in self.columns

                    if _tOrdColAbsent:
                        _genTOrdCol = _genTChunkCol = _genTOrdInChunkCol = _genTDeltaCol = True

                        window = Window \
                            .partitionBy(iCol) \
                            .orderBy(tCol)

                    else:
                        _genTOrdCol = \
                            forceGenTRelAuxCols or \
                            (self.type(self._T_ORD_COL) not in _INT_TYPES)

                        _genTChunkCol = \
                            _genTOrdCol or \
                            _tChunkColAbsent or \
                            (self.type(self._T_CHUNK_COL) not in _INT_TYPES)

                        _genTOrdInChunkCol = \
                            _genTChunkCol or \
                            _tOrdInChunkColAbsent or \
                            (self.type(self._T_ORD_IN_CHUNK_COL) not in _INT_TYPES)

                        _genTDeltaCol = \
                            _genTOrdCol or \
                            _tDeltaColAbsent or \
                            (self.type(self._T_DELTA_COL) != _tDeltaColType)

                        if _genTDeltaCol:
                            window = Window \
                                .partitionBy(iCol) \
                                .orderBy(tCol if _genTOrdCol else self._T_ORD_COL)

                    if arimo.debug.ON:
                        if forceGenTRelAuxCols:
                            self.class_stdout_logger().debug(
                                msg='*** FORCE-GENERATING AUXILIARY COLUMNS {}, {}, {}, {} ***'
                                    .format(self._T_ORD_COL, self._T_CHUNK_COL, self._T_ORD_IN_CHUNK_COL, self._T_DELTA_COL))

                        elif not (_genTOrdCol and _genTChunkCol and _genTOrdInChunkCol and _genTDeltaCol):
                            self.class_stdout_logger().debug(
                                msg='*** SKIP GENERATING ALREADY EXISTING AUXILIARY COLUMN(S) {} ***'
                                    .format((() if _genTOrdCol
                                                else (self._T_ORD_COL,)) +
                                            (() if _genTChunkCol
                                                else (self._T_CHUNK_COL,)) +
                                            (() if _genTOrdInChunkCol
                                                else (self._T_ORD_IN_CHUNK_COL,)) +
                                            (() if _genTDeltaCol
                                                else (self._T_DELTA_COL,))))

                _tComponentExprs = \
                    ((() if _genTHoYCol
                         else (self._T_HoY_COL,)) +

                     (() if _genTQoYCol
                         else (self._T_QoY_COL,)) +

                     (() if _genTMoYCol
                         else (self._T_MoY_COL,)) +

                     # (() if _genTWoYCol
                     #     else (self._T_WoY_COL,)) +

                     # (() if _genTDoYCol
                     #     else (self._T_DoY_COL,)) +

                     (() if _genTPoYCol
                         else (self._T_PoY_COL,)) +

                     (() if _genTQoHCol
                         else (self._T_QoH_COL,)) +

                     (() if _genTMoHCol
                         else (self._T_MoH_COL,)) +

                     (() if _genTPoHCol
                         else (self._T_PoH_COL,)) +

                     (() if _genTMoQCol
                         else (self._T_MoQ_COL,)) +

                     (() if _genTPoQCol
                         else (self._T_PoQ_COL,)) +

                     (() if _genTWoMCol
                         else (self._T_WoM_COL,)) +

                     (() if _genTDoMCol
                      else (self._T_DoM_COL,)) +

                     (() if _genTPoMCol
                         else (self._T_PoM_COL,)) +

                     (() if _genTDoWCol
                         else (self._T_DoW_COL,)) +

                     (() if _genTPoWCol
                         else (self._T_PoW_COL,)) +

                     (((() if _genTHoDCol
                           else (self._T_HoD_COL,)) +

                       (() if _genTPoDCol
                           else (self._T_PoD_COL,)))
                      if _tDailyComponentColsApplicable
                      else ())) \
                    if _tComponentColsApplicable \
                    else ()

                if _castTColType or _genTOrdCol or _genTDeltaCol:
                    self._cache.nPartitions = None

                    if self._cache.nRows is None:
                        if arimo.debug.ON:
                            tic = time.time()

                        self._cache.nRows = self._sparkDF.count()

                        if arimo.debug.ON:
                            toc = time.time()
                            self.class_stdout_logger().debug(
                                msg='*** nRows = {:,}   <{:,.1f} s> ***'.format(self._cache.nRows, toc - tic))
                    
                    if contentCols is None:
                        contentCols = self.contentCols

                    self._sparkDF = \
                        self._sparkDF.select(
                            *(((self._PARTITION_ID_COL,)
                               if self._detPrePartitioned
                               else ()) +

                              (iCol,) +

                              _dColTup +

                              (_tColExpr,

                               sparkSQLFuncs.row_number()
                                .over(window=window)
                                .alias(self._T_ORD_COL)
                               if _genTOrdCol
                               else self._T_ORD_COL) +

                              (() if self._detPrePartitioned or _genTChunkCol
                                  else (self._T_CHUNK_COL,)) +

                              ((sparkSQLFuncs.datediff(
                                    end=_tColExprForDelta,
                                    start=sparkSQLFuncs.lag(
                                            col=_tColExprForDelta,
                                            count=1,
                                            default=None)
                                        .over(window=window))
                                if _tIsDate
                                else (_tColExprForDelta -
                                       sparkSQLFuncs.lag(
                                           col=_tColExprForDelta,
                                           count=1,
                                           default=None)
                                        .over(window=window)))
                               .alias(self._T_DELTA_COL)
                               if _genTDeltaCol
                               else self._T_DELTA_COL,) +

                              (() if self._detPrePartitioned or _genTOrdInChunkCol
                                  else (self._T_ORD_IN_CHUNK_COL,)) +

                              _tComponentExprs +

                              contentCols))

                _schema = self._sparkDF.schema

                _types[self._T_ORD_COL] = _type = _schema[self._T_ORD_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_DELTA_COL] = _type = _schema[self._T_DELTA_COL].dataType.simpleString()
                assert _type == _tDeltaColType

                if self._detPrePartitioned:
                    _firstCols = \
                        (self._PARTITION_ID_COL,
                         iCol) + \
                        _dColTup + \
                        (tCol,
                         self._T_ORD_COL,
                         self._T_DELTA_COL)

                else:
                    if _genTChunkCol:
                        if contentCols is None:
                            contentCols = self.contentCols

                        self._sparkDF = \
                            self._sparkDF.select(
                                iCol,

                                *(_dColTup +

                                  (tCol,
                                   self._T_ORD_COL,

                                   sparkSQLFuncs.expr(
                                      '(({} - 1) DIV {}) + 1'.format(
                                          self._T_ORD_COL,
                                          self._tChunkLen))
                                    .alias(self._T_CHUNK_COL),

                                   self._T_DELTA_COL) +

                                  (() if self._detPrePartitioned or _genTOrdInChunkCol
                                      else (self._T_ORD_IN_CHUNK_COL,)) +

                                  _tComponentExprs +

                                  contentCols))

                    if _genTOrdInChunkCol:
                        self._cache.nPartitions = None

                        if self._cache.nRows is None:
                            if arimo.debug.ON:
                                tic = time.time()

                            self._cache.nRows = self._sparkDF.count()

                            if arimo.debug.ON:
                                toc = time.time()
                                self.stdout_logger.debug(
                                    msg='*** nRows = {:,}   <{:,.1f} s> ***'.format(self._cache.nRows, toc - tic))

                        if contentCols is None:
                            contentCols = self.contentCols

                        self._sparkDF = self._sparkDF \
                            .repartition(
                                iCol,
                                self._T_CHUNK_COL) \
                            .select(
                                iCol,

                                *(_dColTup +

                                  (tCol,

                                   self._T_ORD_COL,
                                   self._T_CHUNK_COL,

                                   sparkSQLFuncs.row_number()
                                    .over(window=Window.partitionBy(iCol, self._T_CHUNK_COL).orderBy(self._T_ORD_COL))
                                    .alias(self._T_ORD_IN_CHUNK_COL),

                                   self._T_DELTA_COL) +

                                  _tComponentExprs +

                                  contentCols))

                    _schema = self._sparkDF.schema

                    _types[self._T_CHUNK_COL] = _type = _schema[self._T_CHUNK_COL].dataType.simpleString()
                    assert _type in _INT_TYPES

                    _types[self._T_ORD_IN_CHUNK_COL] = _type = _schema[self._T_ORD_IN_CHUNK_COL].dataType.simpleString()
                    assert _type in _INT_TYPES

                    _firstCols = \
                        (iCol,) + \
                        _dColTup + \
                        (tCol,
                         self._T_ORD_COL,
                         self._T_CHUNK_COL,
                         self._T_ORD_IN_CHUNK_COL,
                         self._T_DELTA_COL)

            else:
                self.hasTS = False

                if _castTColType:
                    if contentCols is None:
                        contentCols = self.contentCols

                    self._sparkDF = \
                        self._sparkDF.select(
                            *(_dColTup +
                              (_tColExpr,) +
                              contentCols))

                _firstCols = \
                    ((self._PARTITION_ID_COL,)
                     if self._detPrePartitioned
                     else ()) + \
                    _dColTup + (tCol,)

            if _tComponentColsApplicable:
                if (_genTHoYCol or _genTQoYCol or _genTMoYCol or _genTPoYCol or   # _genTWoYCol or _genTDoYCol or
                        _genTWoMCol or _genTDoMCol or _genTPoMCol or
                        _genTDoWCol or _genTPoWCol) or \
                        (_tDailyComponentColsApplicable and
                         (_genTHoDCol or _genTPoDCol)):
                    if contentCols is None:
                        contentCols = self.contentCols

                    self._sparkDF = \
                        self._sparkDF.select(
                            *(_firstCols +
                            
                              (# Half of Year
                               (sparkSQLFuncs.when(sparkSQLFuncs.quarter(tCol) <= 2, 1).otherwise(2)
                                        if _genTQoYCol
                                        else sparkSQLFuncs.expr('IF({} <= 2, 1, 2)'.format(self._T_QoY_COL)))
                                    .alias(self._T_HoY_COL)
                               if _genTHoYCol
                               else self._T_HoY_COL,

                               # Quarter of Year
                               sparkSQLFuncs.quarter(tCol)
                                    .alias(self._T_QoY_COL)
                               if _genTQoYCol
                               else self._T_QoY_COL,

                               # Month of Year
                               sparkSQLFuncs.month(tCol)
                                    .alias(self._T_MoY_COL)
                               if _genTMoYCol
                               else self._T_MoY_COL,

                               # Week of Year
                               # sparkSQLFuncs.weekofyear(tCol)
                               #      .alias(self._T_WoY_COL)
                               # if _genTWoYCol
                               # else self._T_WoY_COL,

                               # Day of Year
                               # sparkSQLFuncs.dayofyear(tCol)
                               #      .alias(self._T_DoY_COL)
                               # if _genTDoYCol
                               # else self._T_DoY_COL,

                               # Part/Proportion/Fraction of Year
                               ((sparkSQLFuncs.month(tCol) / 12)
                                if _genTMoYCol
                                else sparkSQLFuncs.expr('{} / 12'.format(self._T_MoY_COL)))
                                    .alias(self._T_PoY_COL)
                               if _genTPoYCol
                               else self._T_PoY_COL) +

                              (# Quarter of Half-Year
                               () if _genTQoHCol
                                  else (self._T_QoH_COL,)) +

                              (# Month of Half-Year
                               () if _genTMoHCol
                                  else (self._T_MoH_COL,)) +

                              (# Part/Proportion/Fraction of Half-Year
                               () if _genTPoHCol
                                  else (self._T_PoH_COL,)) +

                              (# Month of Quarter
                               () if _genTMoQCol
                                  else (self._T_MoQ_COL,)) +

                              (# Part/Proportion/Fraction of Quarter
                               () if _genTPoQCol
                                  else (self._T_PoQ_COL,)) +

                              (# Week of Month
                               sparkSQLFuncs.least(
                                        (((sparkSQLFuncs.dayofmonth(tCol) - 1) / 7)
                                                .cast(dataType=_INT_TYPE) + 1)
                                            if _genTDoMCol
                                            else sparkSQLFuncs.expr('(({} - 1) DIV 7) + 1'.format(self._T_DoM_COL)),
                                        sparkSQLFuncs.lit(4))
                                    .alias(self._T_WoM_COL)
                               if _genTWoMCol
                               else self._T_WoM_COL,

                               # Day of Month
                               sparkSQLFuncs.dayofmonth(tCol)
                                    .alias(self._T_DoM_COL)
                               if _genTDoMCol
                               else self._T_DoM_COL,

                               # Part/Proportion/Fraction of Month
                               ((sparkSQLFuncs.dayofmonth(tCol)
                                    if _genTDoMCol
                                    else self[self._T_DoM_COL])
                                / sparkSQLFuncs.dayofmonth(sparkSQLFuncs.last_day(tCol)))
                                    .alias(self._T_PoM_COL)
                               if _genTPoMCol
                               else self._T_PoM_COL,

                               # Day of Week as Number (Mon=1, Sun=7)
                               sparkSQLFuncs.date_format(tCol, 'u')
                                    .cast(dataType=_INT_TYPE)
                                    .alias(self._T_DoW_COL)
                               if _genTDoWCol
                               else self._T_DoW_COL,

                               # Part/Proportion/Fraction of Week
                               (sparkSQLFuncs.date_format(tCol, 'u') / 7)
                                    .alias(self._T_PoW_COL)
                               if _genTPoWCol
                               else self._T_PoW_COL) +

                              ((# Hour of Day
                                sparkSQLFuncs.hour(tCol)
                                    .alias(self._T_HoD_COL)
                                if _genTHoDCol
                                else self._T_HoD_COL,

                                # Part/Proportion/Fraction of Day
                                ((sparkSQLFuncs.hour(tCol)
                                        if _genTHoDCol
                                        else self[self._T_HoD_COL]) / 24)
                                    .alias(self._T_PoD_COL)
                                if _genTPoDCol
                                else self._T_PoD_COL)
                               if _tDailyComponentColsApplicable
                               else ()) +

                              contentCols))

                _schema = self._sparkDF.schema

                _types[self._T_HoY_COL] = _type = _schema[self._T_HoY_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_QoY_COL] = _type = _schema[self._T_QoY_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_MoY_COL] = _type = _schema[self._T_MoY_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                # _types[self._T_WoY_COL] = _type = _schema[self._T_WoY_COL].dataType.simpleString()
                # assert _type in _INT_TYPES

                # _types[self._T_DoY_COL] = _type = _schema[self._T_DoY_COL].dataType.simpleString()
                # assert _type in _INT_TYPES

                _types[self._T_PoY_COL] = _type = _schema[self._T_PoY_COL].dataType.simpleString()
                assert _type in _FLOAT_TYPES

                _types[self._T_WoM_COL] = _type = _schema[self._T_WoM_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_DoM_COL] = _type = _schema[self._T_DoM_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_PoM_COL] = _type = _schema[self._T_PoM_COL].dataType.simpleString()
                assert _type in _FLOAT_TYPES

                _types[self._T_DoW_COL] = _type = _schema[self._T_DoW_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_PoW_COL] = _type = _schema[self._T_PoW_COL].dataType.simpleString()
                assert _type in _FLOAT_TYPES

                if _tDailyComponentColsApplicable:
                    _types[self._T_HoD_COL] = _type = _schema[self._T_HoD_COL].dataType.simpleString()
                    assert _type in _INT_TYPES

                    _types[self._T_PoD_COL] = _type = _schema[self._T_PoD_COL].dataType.simpleString()
                    assert _type in _FLOAT_TYPES

                if _genTQoHCol or _genTMoHCol or _genTPoHCol or _genTMoQCol or _genTPoQCol:
                    if contentCols is None:
                        contentCols = self.contentCols

                    self._sparkDF = \
                        self._sparkDF.select(
                            *(_firstCols +

                              (self._T_HoY_COL,
                               self._T_QoY_COL,
                               self._T_MoY_COL,
                               # self._T_WoY_COL,
                               # self._T_DoY_COL,
                               self._T_PoY_COL,

                               # Quarter of Half-Year
                               sparkSQLFuncs.expr('IF({} IN (1, 3), 1, 2)'.format(self._T_QoY_COL))
                                .alias(self._T_QoH_COL)
                               if _genTQoHCol
                               else self._T_QoH_COL,

                               # Month of Half-Year
                               sparkSQLFuncs.when(self[self._T_MoY_COL] <= 6, self[self._T_MoY_COL])
                                .otherwise(self[self._T_MoY_COL] - 6)
                                .alias(self._T_MoH_COL)
                               if _genTMoHCol
                               else self._T_MoH_COL,

                               # Part/Proportion/Fraction of Half-Year
                               ((sparkSQLFuncs.when(self[self._T_MoY_COL] <= 6, self[self._T_MoY_COL])
                                    .otherwise(self[self._T_MoY_COL] - 6)
                                 if _genTMoHCol
                                 else self[self._T_MoH_COL]) / 6)
                                .alias(self._T_PoH_COL)
                               if _genTPoHCol
                               else self._T_PoH_COL,

                               # Month of Quarter
                               sparkSQLFuncs.expr('(({} - 1) % 3) + 1'.format(self._T_MoY_COL))
                                .alias(self._T_MoQ_COL)
                               if _genTMoQCol
                               else self._T_MoQ_COL,

                               # Part/Proportion/Fraction of Quarter
                               ((sparkSQLFuncs.expr('(({} - 1) % 3) + 1'.format(self._T_MoY_COL))
                                 if _genTMoQCol
                                 else self[self._T_MoQ_COL]) / 3)
                                .alias(self._T_PoQ_COL)
                               if _genTPoQCol
                               else self._T_PoQ_COL,

                               self._T_WoM_COL,
                               self._T_DoM_COL,
                               self._T_PoM_COL,

                               self._T_DoW_COL,
                               self._T_PoW_COL) +

                              ((self._T_HoD_COL,
                                self._T_PoD_COL)
                               if _tDailyComponentColsApplicable
                               else ()) +

                              contentCols))

                _schema = self._sparkDF.schema

                _types[self._T_QoH_COL] = _type = _schema[self._T_QoH_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_MoH_COL] = _type = _schema[self._T_MoH_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_PoH_COL] = _type = _schema[self._T_PoH_COL].dataType.simpleString()
                assert _type in _FLOAT_TYPES

                _types[self._T_MoQ_COL] = _type = _schema[self._T_MoQ_COL].dataType.simpleString()
                assert _type in _INT_TYPES

                _types[self._T_PoQ_COL] = _type = _schema[self._T_PoQ_COL].dataType.simpleString()
                assert _type in _FLOAT_TYPES

            if _typesCached:
                self._cache.type.update(_types)
            
        else:
            self.hasTS = False
            self._dCol = None

    def _emptyCache(self):
        self._cache = \
            _Namespace(
                nPartitions=self._cache.nPartitions,
                nRows=self._cache.nRows,

                type=Namespace(**
                    {col: type
                     for col, type in self.dtypes}),

                firstRow=None, aRow=None,

                reprSample=None,

                count={}, distinct={},   # approx.

                nonNullProportion={},   # approx.
                suffNonNullProportionThreshold={},
                suffNonNull={},

                sampleMin={}, sampleMax={}, sampleMean={}, sampleMedian={},
                outlierRstMin={}, outlierRstMax={}, outlierRstMean={}, outlierRstMedian={},
                
                colWidth={})

    def _inheritCache(self, adf, *sameCols, **newColToOldColMappings):
        if adf._cache.nRows:
            if self._cache.nRows is None:
                self._cache.nRows = adf._cache.nRows
            else:
                assert self._cache.nRows == adf._cache.nRows

        commonCols = set(self.columns).intersection(adf.columns)

        if sameCols or newColToOldColMappings:
            for newCol, oldCol in newColToOldColMappings.items():
                assert newCol in self.columns
                assert oldCol in adf.columns

            for sameCol in commonCols.difference(newColToOldColMappings).intersection(sameCols):
                newColToOldColMappings[sameCol] = sameCol

        else:
            newColToOldColMappings = \
                {col: col
                 for col in commonCols}
        
        for cacheCategory in \
                ('count', 'distinct',
                 'nonNullProportion', 'suffNonNullProportionThreshold', 'suffNonNull',
                 'sampleMin', 'sampleMax', 'sampleMean', 'sampleMedian',
                 'outlierRstMin', 'outlierRstMax', 'outlierRstMean', 'outlierRstMedian',
                 'colWidth'):
            for newCol, oldCol in newColToOldColMappings.items():
                if oldCol in adf._cache.__dict__[cacheCategory]:
                    self._cache.__dict__[cacheCategory][newCol] = \
                        adf._cache.__dict__[cacheCategory][oldCol]

    # decorator to up-convert any Spark SQL DataFrame result to SparkADF
    def _decorate(self, obj, nRows=None, **objKwArgs):
        def methodReturningSparkADF(method):
            def decoratedMethod(*methodArgs, **methodKwArgs):
                if 'tCol' in methodKwArgs:
                    tCol_inKwArgs = True
                    tCol = methodKwArgs['tCol']
                else:
                    tCol_inKwArgs = False

                stdKwArgs = self._extractStdKwArgs(methodKwArgs, resetToClassDefaults=False, inplace=False)

                result = method(*methodArgs, **methodKwArgs)

                if isinstance(result, DataFrame):
                    cols = result.columns

                    if self._detPrePartitioned and (self._PARTITION_ID_COL not in cols):
                        stdKwArgs.detPrePartitioned = False
                        stdKwArgs.nDetPrePartitions = None

                    if stdKwArgs.iCol not in cols:
                        stdKwArgs.iCol = self._iCol if self._iCol in cols else self._DEFAULT_I_COL

                    if tCol_inKwArgs:
                        stdKwArgs.tCol = tCol
                    elif self._tCol:
                        stdKwArgs.tCol = self._tCol if self._tCol in cols else self._DEFAULT_T_COL

                    if stdKwArgs.alias and (stdKwArgs.alias == self.alias):
                        stdKwArgs.alias = None
                    
                    return SparkADF(
                        sparkDF=result,
                        nRows=None,   # not sure what the new nRows may be
                        **stdKwArgs.__dict__)

                else:
                    return result

            decoratedMethod.__module__ = method.__module__
            decoratedMethod.__name__ = method.__name__
            decoratedMethod.__doc__ = method.__doc__
            decoratedMethod.__self__ = self   # real SparkADF __self__ instance, which maybe different to method.__self__
            return decoratedMethod

        if callable(obj) and (not isinstance(obj, SparkADF)) and (not isinstance(obj, types.ClassType)):
            return methodReturningSparkADF(method=obj)

        elif isinstance(obj, DataFrame):
            alias = objKwArgs.get('alias')

            if 'tCol' in objKwArgs:
                tCol_inKwArgs = True
                tCol = objKwArgs['tCol']
            else:
                tCol_inKwArgs = False

            stdKwArgs = self._extractStdKwArgs(objKwArgs, resetToClassDefaults=False, inplace=False)

            cols = obj.columns

            if self._detPrePartitioned and (self._PARTITION_ID_COL not in cols):
                stdKwArgs.detPrePartitioned = False
                stdKwArgs.nDetPrePartitions = None

            if stdKwArgs.iCol not in cols:
                stdKwArgs.iCol = self._iCol if self._iCol in cols else self._DEFAULT_I_COL

            if tCol_inKwArgs:
                stdKwArgs.tCol = tCol
            elif self._tCol:
                stdKwArgs.tCol = self._tCol if self._tCol in cols else self._DEFAULT_T_COL

            stdKwArgs.alias = alias

            return SparkADF(sparkDF=obj, nRows=nRows, **stdKwArgs.__dict__)

        else:
            if isinstance(obj, SparkADF) and (nRows is not None):
                if obj._cache.nRows is None:
                    obj._cache.nRows = nRows

                else:
                    assert obj._cache.nRows == nRows, \
                        '{}._decorate(...): Cached nRows {} != Indicated nRows {}'.format(
                            type(self).__name__, obj._cache.nRows, nRows)

            return obj

    def _inplace(self, df, alias=None, iCol=None, tCol=None, tChunkLen=None):
        if isinstance(df, (tuple, list)):   # just in case we're taking in multiple inputs
            df = df[0]

        cols = df.columns

        if isinstance(df, SparkADF):
            isADF = True

            self._sparkDF = df._sparkDF

            if df._detPrePartitioned:
                self._detPrePartitioned = True
                self._nDetPrePartitions = df._nDetPrePartitions

            elif df._PARTITION_ID_COL not in cols:
                self._detPrePartitioned = False
                self._nDetPrePartitions = None

            self._cache.nRows = df._cache.nRows

        elif isinstance(df, DataFrame):
            isADF = False

            self._sparkDF = df

            if self._PARTITION_ID_COL not in cols:
                self._detPrePartitioned = False
                self._nDetPrePartitions = None

            self._cache.nRows = None

        else:
            raise ValueError("*** SparkADF._inplace(...)'s 1st argument must be either SparkADF or Spark SQL DataFrame ***")

        existingICol = self._iCol
        existingTCol = self._tCol

        if iCol in cols:
            self._iCol = iCol
        elif self._iCol not in cols:
            if isADF and df._iCol in cols:
                self._iCol = df._iCol
            elif self._DEFAULT_I_COL in cols:
                self._iCol = self._DEFAULT_I_COL
            else:
                self._iCol = None

        if tCol in cols:
            self._tCol = tCol
        elif self._tCol and (self._tCol not in cols):
            if isADF and df._tCol in cols:
                self._tCol = df._tCol
            elif self._DEFAULT_T_COL in cols:
                self._tCol = self._DEFAULT_T_COL
            else:
                self._tCol = None

        if tChunkLen is not None:
            self._tChunkLen = tChunkLen

        del self._cache.type

        self._organizeTimeSeries(
            forceGenTRelAuxCols=
                self._iCol and self._tCol and
                ((existingICol is None) or (existingICol != self._iCol) or
                 (existingTCol is None) or (existingTCol != self._tCol)))

        self.alias = alias \
            if alias \
            else (self._alias if self._alias or (not isADF)
                              else df._alias)

        if isinstance(df, SparkADF):
            self._cache = df._cache
        else:
            self._emptyCache()

    # **********************
    # PYTHON DEFAULT METHODS
    # __dir__
    # __getattr__
    # __getitem__
    # __repr__
    # __short_repr__

    def __dir__(self):
        return sorted(set(
            dir(type(self)) +
            self.__dict__.keys() +
            dir(DataFrame) +
            dir(self._sparkDF)))

    def __getattr__(self, attr):
        return self._decorate(
            obj=getattr(self._sparkDF, attr),
            nRows=self._cache.nRows)

    def __getitem__(self, item):
        obj = self._sparkDF[item]
        
        return self._decorate(
                obj=obj,
                nRows=self._cache.nRows) \
            if isinstance(item, (list, tuple)) \
          else obj

    def __repr__(self):
        cols_and_types_str = []

        if self._iCol:
            cols_and_types_str += ['(iCol) {}: {}'.format(self._iCol, self._cache.type[self._iCol])]

        if self._dCol:
            cols_and_types_str += ['(dCol) {}: {}'.format(self._dCol, self._cache.type[self._dCol])]

        if self._tCol:
            cols_and_types_str += ['(tCol) {}: {}'.format(self._tCol, self._cache.type[self._tCol])]

        cols_and_types_str += \
            ['{}: {}'.format(col, self._cache.type[col])
             for col in self.contentCols]

        return '{}{:,}-partition{} {}{}{}[{}]'.format(
            '"{}" '.format(self._alias)
                if self._alias
                else '',

            self.nPartitions,

            ' (from {:,} deterministic partitions)'.format(self.nDetPrePartitions)
                if self._detPrePartitioned
                else '',

            '' if self._cache.nRows is None
               else '{:,}-row '.format(self._cache.nRows),

            '(cached) '
                if self.is_cached
                else '',

            type(self).__name__,

            ', '.join(cols_and_types_str))

    @property
    def __short_repr__(self):
        cols_desc_str = []

        if self._iCol:
            cols_desc_str += ['iCol: {}'.format(self._iCol)]

        if self._dCol:
            cols_desc_str += ['dCol: {}'.format(self._dCol)]

        if self._tCol:
            cols_desc_str += ['tCol: {}'.format(self._tCol)]

        cols_desc_str += ['{} content col(s)'.format(len(self.contentCols))]

        return '{}{:,}-partition{} {}{}{}[{}]'.format(
            '"{}" '.format(self._alias)
                if self._alias
                else '',

            self.nPartitions,

            ' (from {:,} deterministic partitions)'.format(self.nDetPrePartitions)
                if self._detPrePartitioned
                else '',

            '' if self._cache.nRows is None
               else '{:,}-row '.format(self._cache.nRows),

            '(cached) '
                if self.is_cached
                else '',

            type(self).__name__,

            ', '.join(cols_desc_str))

    # **********
    # IO METHODS
    # sparkSession
    # create
    # unionAllCols
    # load
    # save

    @property
    def sparkSession(self):
        """
        Underlying ``SparkSession``
        """
        return self._sparkSession

    @classmethod
    @_docstr_adf_kwargs
    def create(cls, data, schema=None, samplingRatio=None, verifySchema=False, sparkConf={}, **kwargs):
        """
        Create ``SparkADF`` from an ``RDD``, a *list* or a ``pandas.DataFrame``

        (*ref:* http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.SparkSession.createDataFrame)

        Return:
            ``SparkADF`` instance

        Args:
            data: ``RDD`` of any kind of *SQL* data representation (e.g. *row*, *tuple*, *int*, *boolean*, ...),
                or *list*, or ``pandas.DataFrame``

            schema: a ``Spark DataType`` or a data type string or a list of column names

            samplingRatio: ratio of rows sampled for type inference
        """
        if not arimo.backend.chkSpark():
            arimo.backend.initSpark(
                sparkConf=sparkConf)

        pandasTSCols = []
        
        if isinstance(data, pandas.DataFrame):
            nRows = len(data)

            for col in data.columns:
                firstItem = data[col].iat[0]
                if isinstance(firstItem, pandas.Timestamp) or (firstItem is pandas.NaT):
                    pandasTSCols.append(col)
                    data[col] = data[col].map(str)

        else:
            nRows = None

        sparkDF = \
            arimo.backend.spark.createDataFrame(
                data=data,
                schema=schema,
                samplingRatio=samplingRatio,
                verifySchema=verifySchema)

        if pandasTSCols:
            sparkDF = sparkDF.selectExpr(
                *(('TIMESTAMP({0}) AS {0}'.format(col)
                   if col in pandasTSCols
                   else col)
                  for col in sparkDF.columns))

            for col in pandasTSCols:
                data[col] = pandas.to_datetime(data[col])

        return SparkADF(sparkDF=sparkDF, nRows=nRows, **kwargs)

    @classmethod
    def unionAllCols(cls, *adfs_and_or_sparkDFs, **kwargs):
        _TMP_TABLE_PREFIX = '_tmp_tbl_'

        _unionRDDs = kwargs.pop('_unionRDDs', False)

        assert all(isinstance(adfs_and_or_sparkDF, (SparkADF, DataFrame))
                   for adfs_and_or_sparkDF in adfs_and_or_sparkDFs)
                
        nDFs = len(adfs_and_or_sparkDFs)

        if nDFs > 1:
            colTypes = {}

            for adf_or_sparkDF in adfs_and_or_sparkDFs:
                for structField in adf_or_sparkDF.schema:
                    if structField.name not in colTypes:
                        colTypes[structField.name] = structField.dataType

            adfs = [(adf_or_sparkDF
                     if isinstance(adf_or_sparkDF, SparkADF)
                     else SparkADF(sparkDF=adf_or_sparkDF))
                    (*((col
                        if col in adf_or_sparkDF.columns
                        else 'NULL AS {}'.format(col))
                       for col in colTypes))
                    for adf_or_sparkDF in adfs_and_or_sparkDFs]

            if _unionRDDs:
                _tmp_table_aliases = ['this']

                for i in range(1, nDFs):
                    adfs[i].alias = '{}{}'.format(_TMP_TABLE_PREFIX, i)
                    _tmp_table_aliases.append(adfs[i].alias)

                return adfs[0](
                    "SELECT \
                        * \
                    FROM \
                        ({})".format(
                        ' UNION ALL '.join(
                            '(SELECT * FROM {})'.format(_tmp_table_alias)
                            for _tmp_table_alias in _tmp_table_aliases)),
                    **kwargs)

            else:
                return SparkADF.create(
                    data=arimo.backend.spark.sparkContext.union(
                            [adf.rdd for adf in adfs]),
                    schema=StructType(
                            [StructField(
                                name=colName,
                                dataType=dataType,
                                nullable=True,
                                metadata=None)
                             for colName, dataType in colTypes.items()]),
                    samplingRatio=None,
                    verifySchema=False,
                    **kwargs)

        else:
            df = adfs_and_or_sparkDFs[0]

            if isinstance(df, SparkADF):
                return df.copy(**kwargs)

            elif isinstance(df, DataFrame):
                return SparkADF(sparkDF=df, nRows=None, **kwargs)

            else:
                raise ValueError('*** All input data frames must be SparkADFs or Spark SQL DataFrames ***')

    @classmethod
    def _test_hdfs_load(cls):
        if not cls._TEST_HDFS_LOAD:
            _TEST_PARQUET_NAME = 'tiny.parquet'

            _TEST_PARQUET_LOCAL_PATH = \
                os.path.join(
                    os.path.dirname(
                        os.path.dirname(arimo.debug.__file__)),
                    'resources',
                    _TEST_PARQUET_NAME)

            assert os.path.isfile(_TEST_PARQUET_LOCAL_PATH)

            _TEST_PARQUET_HDFS_PATH = \
                os.path.join(
                    cls._TMP_DIR_PATH,
                    _TEST_PARQUET_NAME)

            fs.put(
                from_local=_TEST_PARQUET_LOCAL_PATH,
                to_hdfs=_TEST_PARQUET_HDFS_PATH,
                is_dir=False,
                _mv=False)

            cls.load(
                path=_TEST_PARQUET_HDFS_PATH,
                format='parquet',
                verbose=True)
            
            cls._TEST_HDFS_LOAD = True

    @classmethod
    @_docstr_adf_kwargs
    def load(cls, path, format='parquet', schema=None,
             aws_access_key_id=None, aws_secret_access_key=None, verbose=True, sparkConf={}, **options):
        """
        Load/read tabular data into ``SparkADF``

        Return:
            ``SparkADF`` instance

        Args:
            path (str): path to data source

            format (str): one of:

                - ``parquet`` (default)
                - ``orc``
                - ``jdbc``
                - ``json``
                - ``csv``
                - ``text``

            schema (``Spark SQL StructType``): schema of the tabular data

            **options: and any data format-specific loading/reading options
                (*ref:* http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameReader)
        """
        if not arimo.backend.chkSpark():
            arimo.backend.initSpark(
                sparkConf=sparkConf)
        
        if verbose:
            logger = cls.class_stdout_logger()

            msg = 'Loading by {} Format from {}...'.format(
                    format.upper(),
                    path if isinstance(path, _STR_CLASSES)
                         else '{} Paths e.g. {}'.format(len(path), path[:3]))
            logger.info(msg)
            tic = time.time()

        stdKwArgs = \
            {k: options.pop(k, cls._DEFAULT_KWARGS[k])
             for k in cls._DEFAULT_KWARGS}

        format = format.lower()

        if isinstance(path, _STR_CLASSES):
            if path.startswith('s3'):
                if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                    cls._test_hdfs_load()

                    path = s3.s3a_path_with_auth(
                        s3_path=path,
                        access_key_id=aws_access_key_id,
                        secret_access_key=aws_secret_access_key)

                else:
                    _path = tempfile.mkdtemp()

                    s3.sync(
                        from_dir_path=path, to_dir_path=_path,
                        delete=True, quiet=True,
                        access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key)

                    path = _path

            elif format == 'jdbc':
                options['url'] = path

                if path[5:13] == 'postgres':
                    options['driver'] = 'org.postgresql.Driver'

            elif format == 'com.databricks.spark.redshift':
                options['url'] = path

                rds.spark_redshift_options(
                    options,
                    access_key_id=aws_access_key_id,
                    secret_access_key=aws_secret_access_key)

            adf = SparkADF(
                sparkDF=arimo.backend.spark.read.load(
                    path=path,
                    format=format,
                    schema=schema,
                    **options),
                nRows=None,
                **stdKwArgs)

        else:
            _adfs = []

            for _path in path:
                _adf = SparkADF.load(
                        path=_path,
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        format=format,
                        schema=schema,
                        verbose=False,
                        **options)

                _adfs.append(_adf)

            adf = SparkADF.unionAllCols(
                *_adfs,
                **stdKwArgs)

        if verbose:
            toc = time.time()
            logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

        return adf

    def save(self, path, format='parquet',
             aws_access_key_id=None, aws_secret_access_key=None,
             mode='overwrite', partitionBy=None,
             verbose=True, switch=False, **options):
        """
        Save/write ``SparkADF``'s content to permanent storage

        Args:
            path (str): path to which to save data

            format (str): one of:

                - ``parquet`` (default)
                - ``orc``
                - ``jdbc``
                - ``json``
                - ``csv``
                - ``text``

            mode (str): behavior when data already exists at specified path:

                - ``append``: append ``SparkADF``'s content to existing data

                - ``overwrite``: overwrite existing data

                - ``error``: throw error/exception

                - ``ignore``: silently ignore & skip

            partitionBy: names of partitioning columns

            **options: format-specific loading/reading options
                (*ref:* http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrameWriter)
        """
        if ('compression' in options) and (options['compression'] is None):
            options['compression'] = 'none'

        sparkDF = self._sparkDF[self.indexCols + self.contentCols].drop(self._PARTITION_ID_COL)

        if (partitionBy is None) and self._dCol and (self._dCol != self._tCol):
            partitionBy = self._dCol

        if verbose:
            msg = 'Saving Columns {} by {} Format{}{} in {} Mode to "{}"{}...' \
                    .format(
                        sparkDF.columns,
                        format.upper(),
                        ' (Compression: {})'.format(options['compression'])
                            if 'compression' in options
                            else '',
                        ', Partitioned by {},'.format(
                                '"{}"'.format(partitionBy)
                                if isinstance(partitionBy, _STR_CLASSES)
                                else partitionBy)
                            if partitionBy
                            else '',
                        mode.upper(),
                        path,
                        ' (DB Table "{}")'.format(options['dbtable'])
                            if 'dbtable' in options
                            else '')
            self.stdout_logger.info(msg)
            tic = time.time()

        format = format.lower()

        if format == 'jdbc':
            options['url'] = path

            if path[5:13] == 'postgres':
                options['driver'] = 'org.postgresql.Driver'

        elif format == 'com.databricks.spark.redshift':
            options['url'] = path

            rds.spark_redshift_options(
                options,
                access_key_id=aws_access_key_id,
                secret_access_key=aws_secret_access_key)

        if path.startswith('s3'):
            if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                self._test_hdfs_load()

                if options.pop('getToLocal', True):   # *** HDFS-to-S3 transfers are SLOW ***
                    _path = tempfile.mkdtemp()

                    sparkDF.write.save(
                        path=_path,
                        format=format,
                        mode=mode,
                        partitionBy=partitionBy,
                        **options)

                    fs.get(
                        from_hdfs=_path, to_local=_path,
                        is_dir=True, overwrite=True, _mv=True,
                        must_succeed=True)

                    s3.sync(
                        from_dir_path=_path, to_dir_path=path,
                        delete=(mode == 'overwrite'), quiet=True,
                        access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key)

                    fs.rm(path=_path,
                          hdfs=False,
                          is_dir=True)

                else:
                    path = s3.s3a_path_with_auth(
                        s3_path=path,
                        access_key_id=aws_access_key_id,
                        secret_access_key=aws_secret_access_key)

                    sparkDF.cache()

                    nTries = options.pop('nTries', 3)
                    err = None

                    while nTries > 0:
                        try:
                            sparkDF.write.save(
                                path=path,
                                format=format,
                                mode=mode,
                                partitionBy=partitionBy,
                                **options)

                            break

                        except Exception as err:
                            nTries -= 1

                            self.stdout_logger.warning(msg + ' *** RETRYING ({} TRIES REMAINING)... ***'.format(nTries))

                    if not nTries:
                        raise err

                    sparkDF.unpersist()

            else:
                _path = tempfile.mkdtemp()

                sparkDF.write.save(
                    path=_path,
                    format=format,
                    mode=mode,
                    partitionBy=partitionBy,
                    **options)

                s3.sync(
                    from_dir_path=_path, to_dir_path=path,
                    delete=(mode == 'overwrite'), quiet=True,
                    access_key_id=aws_access_key_id, secret_access_key=aws_secret_access_key)

                fs.rm(path=_path,
                      hdfs=False,
                      is_dir=True)

        elif format == 'com.databricks.spark.redshift':
            assert mode != 'overwrite'
            if mode == 'OVERWRITE':
                mode = 'overwrite'

            sparkDF.cache()

            sparkDF.write.save(
                path=path,
                format=format,
                mode=mode,
                partitionBy=partitionBy,
                **options)

            sparkDF.unpersist()

        else:
            if format == 'jdbc':
                assert mode != 'overwrite'
                if mode == 'OVERWRITE':
                    mode = 'overwrite'

            sparkDF.write.save(
                path=path,
                format=format,
                mode=mode,
                partitionBy=partitionBy,
                **options)

        if switch:   # then use the newly-saved file as SparkADF's source
            assert format not in ('jdbc', 'com.databricks.spark.redshift')
            assert mode == 'overwrite'

            self._inplace(
                SparkADF.load(
                    path=path,
                    format=format,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    verbose=verbose,
                    nRows=self._cache.nRows))

        if verbose:
            toc = time.time()
            self.stdout_logger.info(msg + ' done!   <{:,.1f} m>'.format((toc - tic) / 60))

    # *************************
    # SPARK PERSISTENCE METHODS
    # cache
    # checkpoint

    def cache(self, eager=True, verbose=True):
        if arimo.debug.ON:
            eager = verbose = True
        elif not eager:
            verbose = False

        if verbose:
            self.stdout_logger.info('Caching Columns {}...'.format(self.columns))
            tic = time.time()

        if self._alias:
            assert not arimo.backend.spark.catalog.isCached(tableName=self._alias), \
                '*** Spark SQL Table "{}" Already Cached ***'.format(self._alias)

            arimo.backend.spark.catalog.cacheTable(tableName=self._alias)

        self._sparkDF.cache()

        if eager:
            self._cache.nRows = self._sparkDF.count()

            if verbose:
                toc = time.time()
                self.stdout_logger.info('Cached!   <{:,.1f} m>'.format((toc - tic) / 60))

    def checkpoint(self, cache=False, format=None, eager=True, verbose=True, **options):
        if cache:
            self.cache(
                eager=eager,
                verbose=verbose)

        if arimo.debug.ON:
            eager = verbose = True
        elif not eager:
            verbose = False

        if verbose:
            msg = 'Checkpointing...'
            self.stdout_logger.info(msg)
            tic = time.time()

        if format is None:
            self._sparkDF.checkpoint(eager=eager)

        else:
            self.save(
                path=os.path.join(
                    arimo.backend._SPARK_CKPT_DIR,
                    '{}.{}'.format(
                        uuid.uuid4(),
                        format)),
                format=format,
                mode='overwrite',
                verbose=verbose,
                switch=True,
                **options)

        if verbose:
            toc = time.time()
            self.stdout_logger.info(msg + ' done!   <{:,.1f} m>'.format((toc - tic) / 60))

    # *************************
    # KEY (SETTABLE) PROPERTIES
    # sparkDF
    # alias
    # nPartitions
    # detPrePartitioned
    # nDetPrePartitions
    # _maxPartitionId
    # iCol
    # tCol
    # tChunkLen
    # _assignReprSample

    @property
    @_docstr_settable_property
    def sparkDF(self):
        """
        Underlying ``Spark DataFrame``
        """
        return self._sparkDF

    @sparkDF.setter
    def sparkDF(self, sparkDF):
        self._inplace(df=sparkDF)

    @property
    @_docstr_settable_property
    def alias(self):
        """
        Name of ``SparkADF`` in the ``SparkSession`` *(str, default = None)*
        """
        return self._alias

    @alias.setter
    def alias(self, alias):
        # *** DON'T DO THE BELOW BECAUSE IT UNCACHES THE DATAFRAME IF THE DATAFRAME IS CACHED ***
        # if self._alias:
        #     arimo.backend.spark.catalog.dropTempView(viewName=self._alias)

        self._alias = alias
        if alias:
            self.createOrReplaceTempView(name=alias)

    @alias.deleter
    def alias(self):
        self.alias = None

    @property
    def nPartitions(self):
        if self._cache.nPartitions is None:
            self._cache.nPartitions = self._sparkDF.rdd.getNumPartitions()
        return self._cache.nPartitions

    @property
    def detPrePartitioned(self):
        return self._detPrePartitioned

    @detPrePartitioned.setter
    def detPrePartitioned(self, detPrePartitioned):
        if detPrePartitioned != self._detPrePartitioned:
            self._detPrePartitioned = detPrePartitioned

            if detPrePartitioned:
                assert self._PARTITION_ID_COL not in self.columns
                self.withColumn(
                    colName=self._PARTITION_ID_COL,
                    col=sparkSQLFuncs.spark_partition_id(),
                    detPrePartitioned=True,
                    inplace=True)

            else:
                self.drop(
                    self._PARTITION_ID_COL,
                    detPrePartitioned=False,
                    inplace=True)

    @detPrePartitioned.deleter
    def detPrePartitioned(self):
        self.detPrePartitioned = False

    @property
    def nDetPrePartitions(self):
        return self._nDetPrePartitions

    @property
    def _maxPartitionId(self):
        return self._sparkDF.select(
            sparkSQLFuncs.max(self._PARTITION_ID_COL)) \
            .first()[0]

    @property
    @_docstr_settable_property
    def iCol(self):
        """
        Name of ``SparkADF``'s *identity*/*entity* column *(str, default = 'id')*
        """
        return self._iCol

    @iCol.setter
    def iCol(self, iCol):
        assert (iCol is None) or (iCol in self.columns), \
            '*** {} Not Among {} ***'.format(iCol, self.columns)

        if iCol != self._iCol:
            self._iCol = iCol
            self._organizeTimeSeries(forceGenTRelAuxCols=True)
            if self.hasTS:   # if newly turned into a time-series SparkADF, then set alias to update underlying table
                self.alias = self._alias
                self._cache.reprSample = None

    @iCol.deleter
    def iCol(self):
        self.iCol = None

    @property
    @_docstr_settable_property
    def tCol(self):
        """
        Name of ``SparkADF``'s *timestamp* column *(str)*
        """
        return self._tCol

    @tCol.setter
    def tCol(self, tCol):
        assert (tCol is None) or (tCol in self.columns), \
            '*** {} Not Among {} ***'.format(tCol, self.columns)

        if tCol != self._tCol:
            self._tCol = tCol
            self._organizeTimeSeries(forceGenTRelAuxCols=True)
            self.alias = self._alias   # set alias to update underlying table
            self._cache.reprSample = None

    @tCol.deleter
    def tCol(self):
        self.tCol = None

    @property
    @_docstr_settable_property
    def tChunkLen(self):
        """
        Max size of time-series chunks per `id`
        """
        assert not self._detPrePartitioned, \
            '*** {}.tChunkLen NOT APPLICABLE WITH PRE-PARTITIONED SparkADF ***'.format(type(self))
        
        return self._tChunkLen

    @tChunkLen.setter
    def tChunkLen(self, tChunkLen):
        assert not self._detPrePartitioned, \
            '*** {}.tChunkLen NOT APPLICABLE WITH PRE-PARTITIONED SparkADF ***'.format(type(self))

        if tChunkLen != self._tChunkLen:
            self._tChunkLen = tChunkLen

            if self.hasTS:
                self._sparkDF = self._sparkDF \
                    .select(
                        self._iCol,
                        self._tCol,
                        self._T_ORD_COL,

                        sparkSQLFuncs.expr(
                            '(({} - 1) DIV {}) + 1'.format(
                                self._T_ORD_COL,
                                tChunkLen))
                            .alias(self._T_CHUNK_COL),

                        self._T_DELTA_COL,

                        *(self.tComponentAuxCols +

                          self.contentCols)) \
                    .repartition(
                        self._iCol,
                        self._T_CHUNK_COL) \
                    .select(
                        self._iCol,
                        self._tCol,
                        self._T_ORD_COL,
                        self._T_CHUNK_COL,

                        sparkSQLFuncs.row_number()
                            .over(window=Window.partitionBy(self._iCol, self._T_CHUNK_COL).orderBy(self._T_ORD_COL))
                            .alias(self._T_ORD_IN_CHUNK_COL),

                        self._T_DELTA_COL,

                        *(self.tComponentAuxCols +

                          self.contentCols))

                # set alias to update underlying table
                self.alias = self._alias

    def _assignReprSample(self):
        adf = self.sample(
                n=self._reprSampleSize,
                anon=True) \
            .repartition(
                1,
                alias=(self.alias + self._REPR_SAMPLE_ALIAS_SUFFIX)
                    if self.alias
                    else None)

        adf.cache(
            eager=True,
            verbose=True)

        self._reprSampleSize = adf.nRows

        self._cache.reprSample = adf

        self._cache.nonNullProportion = {}
        self._cache.suffNonNull = {}

    # *********************
    # ROWS, COLUMNS & TYPES
    # __len__ / nRows / nrow
    # nCols / ncol
    # shape / dim
    # colNames / colnames / names
    # types / type / typeIsNum / typeIsComplex
    # metadata

    @property
    def nRows(self):
        # Number of rows
        if self._cache.nRows is None:
            self._cache.nRows = self._sparkDF.count()
        return self._cache.nRows

    @property
    def types(self):
        """
        *dict* of data type per column name
        """
        if not hasattr(self._cache, 'type'):
            self._cache.type = Namespace(**dict(self.dtypes))

        return self._cache.type

    def type(self, col):
        """
        Return:
            Type of a column

        Args:
            col (str): name of column
        """
        return self.types[col]

    def typeIsNum(self, col):
        t = self.type(col)

        return t.startswith(_DECIMAL_TYPE_PREFIX) \
            or (t in _NUM_TYPES)

    def typeIsComplex(self, col):
        t = self.type(col)

        return t.startswith(_ARRAY_TYPE_PREFIX) \
            or t.startswith(_MAP_TYPE_PREFIX) \
            or t.startswith(_STRUCT_TYPE_PREFIX)

    def metadata(self, *cols):
        if not cols:
            cols = self.contentCols

        return Namespace(**
                {col: Namespace(**self._sparkDF._schema[str(col)].metadata)
                 for col in cols}) \
            if len(cols) > 1 \
            else Namespace(**self._sparkDF._schema[str(cols[0])].metadata)

    # *************
    # COLUMN GROUPS
    # indexCols
    # tRelAuxCols
    # possibleFeatureContentCols
    # possibleCatContentCols

    @property
    def indexCols(self):
        cols = self.columns
        return \
            ((self._PARTITION_ID_COL,) if self._detPrePartitioned and (self._PARTITION_ID_COL in cols) else ()) + \
            ((self._iCol,) if self._iCol in cols else ()) + \
            ((self._dCol,) if (self._dCol in cols) and (self._dCol != self._tCol) else ()) + \
            ((self._tCol,) if self._tCol in cols else ())

    @property
    def tRelAuxCols(self):
        return ((self._T_ORD_COL, self._T_DELTA_COL)
                if self._detPrePartitioned
                else self._T_REL_AUX_COLS) \
            if self.hasTS \
          else ()

    @property
    def possibleFeatureContentCols(self):
        chk = lambda t: \
            (t == _BOOL_TYPE) or \
            (t == _STR_TYPE) or \
            t.startswith(_DECIMAL_TYPE_PREFIX) or \
            (t in _FLOAT_TYPES) or \
            (t in _INT_TYPES)

        return tuple(
            col for col in self.contentCols
                if chk(self.type(col)))

    @property
    def possibleCatContentCols(self):
        return tuple(
            col for col in self.contentCols
                if self.type(col).startswith(_DECIMAL_TYPE_PREFIX) or self.type(col) in _POSSIBLE_CAT_TYPES)

    # **********
    # SQL & SHOW
    # copy
    # select
    # sql
    # __call__
    # show

    @_docstr_adf_kwargs
    def copy(self, **kwargs):
        """
        Return:
            A copy of the ``SparkADF``
        """
        adf = self._decorate(
            obj=self._sparkDF,
            nRows=self._cache.nRows,
            **kwargs)

        adf._inheritCache(self)

        return adf

    @_docstr_adf_kwargs
    def select(self, *exprs, **kwargs):
        """
        Return:
            ``SparkADF`` that is ``SELECT``'ed according to the given expressions

        Args:
            *exprs: series of strings or ``Spark SQL`` expressions
        """
        if exprs:
            inheritCache = kwargs.pop('inheritCache', '*' in exprs)

        else:
            exprs = '*',
            inheritCache = kwargs.pop('inheritCache', True)

        inheritNRows = kwargs.pop('inheritNRows', inheritCache)

        adf = self._decorate(
            obj=self._sparkDF.selectExpr(*exprs)
                if all(isinstance(expr, _STR_CLASSES) for expr in exprs)
                else self._sparkDF.select(*exprs),
            nRows=self._cache.nRows
                if inheritNRows
                else None,
            **kwargs)

        if inheritCache:
            adf._inheritCache(self)

        return adf

    @_docstr_adf_kwargs
    def sql(self, query='SELECT * FROM this', tempAlias='this', **kwargs):
        """
        Return:
            ``SparkADF`` that is ``SELECT``'ed according to the given query

        Args:
            query (str, default = 'SELECT * FROM this'): *SQL* query beginning with "``SELECT``"

        Keyword Args:
            tempAlias (str, default = 'this'): name of temporary *SQL* view to refer to original ``SparkADF``
        """
        origAlias = self._alias
        self.alias = tempAlias

        try:
            _lower_query = query.strip().lower()
            assert _lower_query.startswith('select')

            sparkDF = arimo.backend.spark.sql(query)
            self.alias = origAlias

            inheritCache = \
                kwargs.pop(
                    'inheritCache',
                    (('select *' in _lower_query) or ('select {}.*'.format(tempAlias.lower()) in _lower_query)) and
                    ('where' not in _lower_query) and ('join' not in _lower_query) and ('union' not in _lower_query))

            inheritNRows = kwargs.pop('inheritNRows', inheritCache)

            adf = self._decorate(
                obj=sparkDF,
                nRows=self._cache.nRows
                    if inheritNRows
                    else None,
                **kwargs)

            if inheritCache:
                adf._inheritCache(self)

            return adf

        except Exception as exception:
            self.alias = origAlias
            raise exception

    @_docstr_adf_kwargs
    def __call__(self, *args, **kwargs):
        """
        (**SYNTACTIC SUGAR**)

        Return:
            ``SparkADF`` instance

        There are 2 uses of this convenience method:

            - ``adf(func, *args, **kwargs)``: equivalent to ``func(adf, *args, **kwargs)``,
                    with the final result being converted to ``SparkADF`` if it is a ``Spark DataFrame``

            - ``adf(*exprs, **kwargs)``: invokes *SQL* ``SELECT`` query with ``*exprs`` being either:

                - series of strings or ``Spark SQL`` expressions (equivalent to ``adf.select(*exprs)``); or

                - 1 single *SQL* query beginning with "``SELECT``" (equivalent to ``adf.sql(query)``)
        """
        if args:
            arg = args[0]

            if callable(arg) and (not isinstance(arg, SparkADF)) and (not isinstance(arg, types.ClassType)):
                args = args[1:] \
                    if (len(args) > 1) \
                    else ()

                stdKwArgs = self._extractStdKwArgs(kwargs, resetToClassDefaults=False, inplace=False)

                if stdKwArgs.alias and (stdKwArgs.alias == self.alias):
                    stdKwArgs.alias = None

                inheritCache = \
                    kwargs.pop(
                        'inheritCache',
                        (getattr(arg, '__name__') == 'transform') and
                        isinstance(getattr(arg, '__self__'), Transformer))

                inheritNRows = kwargs.pop('inheritNRows', inheritCache)

                adf = self._decorate(
                    obj=arg(self, *args, **kwargs),
                    nRows=self._cache.nRows
                        if inheritNRows
                        else None,
                    **stdKwArgs.__dict__)

                if inheritCache:
                    adf._inheritCache(self)

                return adf

            elif (len(args) == 1) and isinstance(arg, _STR_CLASSES) and arg.strip().lower().startswith('select'):
                return self.sql(query=arg, **kwargs)

            else:
                return self.select(*args, **kwargs)

        else:
            return self.sql(**kwargs)

    @_docstr_verbose
    def show(self, *exprs, **kwargs):
        """
        Display the result of *SQL* ``SELECT`` query

        Args:
            *exprs: either:

                - series of strings or ``Spark SQL`` expressions; or

                - 1 single *SQL* query beginning with "``SELECT``"

            **kwargs:

                - **n** *(int)*: number of rows to show
        """
        printSchema = kwargs.pop('schema', False)
        __tAuxCols__ = kwargs.pop('__tAuxCols__', False)

        print(self)

        adf = self(*exprs) \
            if exprs \
            else self

        sparkDF = \
            adf._sparkDF[
                adf.indexCols +

                ((adf.tAuxCols
                  if __tAuxCols__
                  else adf.tRelAuxCols)

                 if adf.hasTS

                 else (tuple(col for col in adf._T_AUX_COLS
                             if col in adf.columns)
                       if __tAuxCols__
                       else ())) +

                adf.contentCols]

        if printSchema:
            sparkDF.printSchema()

        sparkDF.show(**kwargs)

    # ****************
    # FIRST / REPR ROW
    # first / firstRow
    # aRow
    # _colWidth

    @property
    def first(self):
        if self._cache.firstRow is None:
            if arimo.debug.ON:
                tic = time.time()

            self._cache.firstRow = row = self._sparkDF.first()

            if arimo.debug.ON:
                toc = time.time()
                self.stdout_logger.debug(
                    msg='*** FIRST ROW OF COLUMNS {}: {}   <{:,.1f} s> ***'
                        .format(self.columns, row, toc - tic))

        return self._cache.firstRow

    @property
    def firstRow(self):
        return self.first

    @property
    def aRow(self):
        if self._cache.aRow is None:
            self._cache.aRow = \
                (self.first
                 if self._cache.reprSample is None
                 else self._cache.reprSample.first) \
                if self._cache.firstRow is None \
                else self._cache.firstRow

        return self._cache.aRow

    def _colWidth(self, *cols, **kwargs):   # *** NOT APPLICABLE TO COLUMNS PRODUCED BY COLLECT_LIST OVER WINDOW ***
        asDict = kwargs.pop('asDict', False)

        for col in set(cols).difference(self._cache.colWidth):
            colType = self.type(col)

            assert not (colType.startswith(_ARRAY_TYPE_PREFIX) or
                        colType.startswith(_MAP_TYPE_PREFIX) or
                        colType.startswith(_STRUCT_TYPE_PREFIX)), \
                '*** {}._colWidth(<AComplexColumn>) MUST BE MANUALLY CACHED ***'

            if colType == _VECTOR_TYPE:
                try:
                    self._cache.colWidth[col] = \
                        self.metadata(col).ml_attr.num_attrs

                except Exception as err:
                    print('*** VECTOR COLUMN "{}" IN SCHEMA {} ***'.format(col, self._sparkDF._schema))
                    raise err

            else:
                self._cache.colWidth[col] = 1

        return Namespace(**
                {col: self._cache.colWidth[col]
                 for col in cols}) \
            if (len(cols) > 1) or asDict \
          else self._cache.colWidth[cols[0]]

    # ****************
    # COLUMN PROFILING
    # _nonNullCol
    # count
    # nonNullProportion
    # distinct
    # quantile
    # sampleStat / sampleMedian
    # outlierRstStat / outlierRstMin / outlierRstMax / outlierRstMedian
    # profile

    @lru_cache()
    def _nonNullCol(self, col, lower=None, upper=None, strict=False):
        colType = self.type(col)

        condition = \
            '({} IS NOT NULL)'.format(col) + \
            ('' if colType.startswith(_ARRAY_TYPE_PREFIX) or
                   colType.startswith(_MAP_TYPE_PREFIX) or
                   colType.startswith(_STRUCT_TYPE_PREFIX)
                else " AND (STRING({}) != 'NaN')".format(col))

        if (colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _NUM_TYPES)) and \
                (pandas.notnull(lower) or pandas.notnull(upper)):
            _debugLogCondition = False

            if colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _FLOAT_TYPES):
                numStrFormatter = '%.9f'

                if pandas.notnull(lower) and pandas.notnull(upper) and (lower + 1e-6 > upper):
                    self.stdout_logger.warning(
                        msg='*** LOWER {} >= UPPER {} ***'
                            .format(lower, upper))

                    upper = lower + 1e-6

                    _debugLogCondition = True

            else:
                numStrFormatter = '%i'

            equalSignStr = '' if strict else '='

            condition += ' AND {}{}{}'.format(
                '' if pandas.isnull(lower)
                   else '({} >{} {})'.format(col, equalSignStr, numStrFormatter % lower),

                '' if pandas.isnull(lower) or pandas.isnull(upper)
                   else ' AND ',

                '' if pandas.isnull(upper)
                   else '({} <{} {})'.format(col, equalSignStr, numStrFormatter % upper))

            if _debugLogCondition:
                self.stdout_logger.debug(
                    msg='*** CONDITION: "{}" ***'
                        .format(condition))

        return self._sparkDF[[col]].filter(condition=condition)

    @_docstr_verbose
    def count(self, *cols, **kwargs):
        """
        Return:
            - If 1 column name is given, return its corresponding non-``NULL`` count

            - If multiple column names are given, return a {``col``: corresponding non-``NULL`` count} *dict*

            - If no column names are given, return a {``col``: corresponding non-``NULL`` count} *dict* for all columns

        Args:
             *cols (str): column name(s)

             **kwargs:
        """
        if not cols:
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.count(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if col not in self._cache.count:
                verbose = True \
                    if arimo.debug.ON \
                    else kwargs.get('verbose')

                if verbose:
                    tic = time.time()

                self._cache.count[col] = result = \
                    self._sparkDF.select(sparkSQLFuncs.count(col)).first()[0] \
                        if self.typeIsComplex(col) \
                        else self._nonNullCol(col=col).count()

                assert isinstance(result, int), \
                    '*** "{}" COUNT = {} ***'.format(col, result)

                if verbose:
                    toc = time.time()
                    self.stdout_logger.info(
                        msg='No. of Non-NULLs of Column "{}" = {:,}   <{:,.1f} s>'
                            .format(col, result, toc - tic))

            return self._cache.count[col]

    @_docstr_verbose
    def nonNullProportion(self, *cols, **kwargs):
        """
        Return:
            - If 1 column name is given, return its *approximate* non-``NULL`` proportion

            - If multiple column names are given, return {``col``: approximate non-``NULL`` proportion} *dict*

            - If no column names are given, return {``col``: approximate non-``NULL`` proportion} *dict* for all columns

        Args:
             *cols (str): column name(s)

             **kwargs:
        """
        if not cols:
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.nonNullProportion(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if col not in self._cache.nonNullProportion:
                self._cache.nonNullProportion[col] = \
                    self.reprSample.count(col, **kwargs) / self.reprSampleSize

            return self._cache.nonNullProportion[col]

    @_docstr_verbose
    def distinct(self, *cols, **kwargs):
        """
        Return:
            *Approximate* list of distinct values of ``SparkADF``'s column ``col``,
                with optional descending-sorted counts for those values

        Args:
            col (str): name of a column

            count (bool): whether to count the number of appearances of each distinct value of the specified ``col``

            collect (bool): whether to return a ``pandas.DataFrame`` (``collect=True``) or a ``Spark SQL DataFrame``

            **kwargs:
        """
        if not cols:
            cols = self.contentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.distinct(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            count = kwargs.get('count', True)

            collect = kwargs.get('collect', True)

            if col in self._cache.distinct:
                series = self._cache.distinct[col]

                assert isinstance(series, pandas.Series)

                if (series.dtype in (float, int)) or not count:
                    return series

            verbose = True \
                if arimo.debug.ON \
                else kwargs.get('verbose')

            if verbose:
                msg = 'Profiling Distinct Values of Column "{}"...'.format(col)
                self.stdout_logger.info(msg)
                tic = time.time()

            adf = self.reprSample(
                    'SELECT \
                        {0}, \
                        (COUNT(*) / {1}) AS __proportion__ \
                    FROM \
                        this \
                    GROUP BY \
                        {0} \
                    ORDER BY \
                        __proportion__ DESC'
                        .format(col, self.reprSampleSize),
                    **kwargs) \
                if count \
                else \
                    self.reprSample(
                        'SELECT \
                            DISTINCT({}) \
                        FROM \
                            this'.format(col),
                        **kwargs)

            if collect:
                df = adf.toPandas()

                dups = {k: v
                        for k, v in Counter(df[col]).items()
                        if v > 1}

                if dups:
                    assert all(pandas.isnull(k) for k in dups), \
                        '*** {}.distinct("{}"): POSSIBLE SPARK SQL/HIVEQL BUG: DUPLICATES {} ***'.format(self, col, dups)

                    index_of_first_row_with_null = None
                    row_indices_to_delete = []

                    for i, row in df.iterrows():
                        if pandas.isnull(row[col]):
                            if index_of_first_row_with_null is None:
                                index_of_first_row_with_null = i

                            else:
                                row_indices_to_delete.append(i)

                                if count:
                                    df.at[index_of_first_row_with_null, '__sample_count__'] += df.at[i, '__sample_count__']
                                    df.at[index_of_first_row_with_null, '__proportion__'] += df.at[i, '__proportion__']

                    df.drop(
                        index=row_indices_to_delete,
                        level=None,
                        inplace=True,
                        errors='raise')

                    if count:
                        df.sort_values(
                            by='__proportion__',
                            ascending=False,
                            inplace=True,
                            kind='quicksort',
                            na_position='last')

                        df.reset_index(
                            level=None,
                            drop=True,
                            inplace=True,
                            col_level=0,
                            col_fill='')

                self._cache.distinct[col] = \
                    result = \
                        df.set_index(
                            keys=col,
                            drop=True,
                            append=False,
                            inplace=False,
                            verify_integrity=False).__proportion__ \
                        if count \
                        else df[col]

            else:
                result = adf

            if verbose:
                toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

            return result

    @lru_cache()
    def quantile(self, *cols, **kwargs):   # make Spark SQL approxQuantile method NULL-resistant
        if len(cols) > 1:
            return Namespace(**
                {col: self.quantile(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            q = kwargs.get('q', .5)
            _multiQs = isinstance(q, (list, tuple))

            relErr = kwargs.get('relativeError', 0.)

            if self.count(col):
                result = \
                    self._nonNullCol(col=col) \
                        .approxQuantile(
                            col=col,
                            probabilities=q
                                if _multiQs
                                else (q,),
                            relativeError=relErr)

                return result \
                    if _multiQs \
                  else result[0]

            else:
                return len(q) * [numpy.nan] \
                    if _multiQs \
                  else numpy.nan

    @_docstr_verbose
    def sampleStat(self, *cols, **kwargs):
        """
        *Approximate* measurements of a certain statistic on **numerical** columns

        Args:
            *cols (str): column name(s)

            **kwargs:

                - **stat**: one of the following:

                    - ``avg``/``mean`` (default)
                    - ``median``
                    - ``min``
                    - ``max``
        """
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.sampleStat(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                stat = kwargs.pop('stat', 'mean').lower()
                if stat == 'avg':
                    stat = 'mean'
                capitalizedStatName = stat.capitalize()
                s = 'sample{}'.format(capitalizedStatName)

                if hasattr(self, s):
                    return getattr(self, s)(col, **kwargs)

                else:
                    if s not in self._cache:
                        setattr(self._cache, s, {})
                    cache = getattr(self._cache, s)

                    if col not in cache:
                        verbose = True \
                            if arimo.debug.ON \
                            else kwargs.get('verbose')

                        if verbose:
                            tic = time.time()

                        cache[col] = result = \
                            self.reprSample \
                                ._nonNullCol(col=col) \
                                .select(getattr(sparkSQLFuncs, stat)(col)) \
                                .first()[0]

                        assert isinstance(result, (float, int)), \
                            '*** "{}" SAMPLE {} = {} ***'.format(col, capitalizedStatName.upper(), result)

                        if verbose:
                            toc = time.time()
                            self.stdout_logger.info(
                                msg='Sample {} for Column "{}" = {:,.3g}   <{:,.1f} s>'
                                    .format(capitalizedStatName, col, result, toc - tic))

                    return cache[col]

            else:
                raise ValueError(
                    '{0}.sampleStat({1}, ...): Column "{1}" Is Not of Numeric Type'
                        .format(self, col))

    def sampleMedian(self, *cols, **kwargs):
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.sampleMedian(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                if 'sampleMedian' not in self._cache:
                    self._cache.sampleMedian = {}

                if col not in self._cache.sampleMedian:
                    verbose = True \
                        if arimo.debug.ON \
                        else kwargs.get('verbose')

                    if verbose:
                        tic = time.time()

                    self._cache.sampleMedian[col] = result = \
                        self.reprSample \
                            .quantile(
                                col,
                                q=.5,
                                relativeError=0.)

                    assert isinstance(result, (float, int)), \
                        '*** "{}" SAMPLE MEDIAN = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Sample Median of Column "{}" = {:,.3g}   <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.sampleMedian[col]

            else:
                raise ValueError(
                    '{0}.sampleMedian({1}, ...): Column "{1}" Is Not of Numeric Type'
                        .format(self, col))

    def outlierRstStat(self, *cols, **kwargs):
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.outlierRstStat(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                stat = kwargs.pop('stat', 'mean').lower()
                if stat == 'avg':
                    stat = 'mean'
                capitalizedStatName = stat.capitalize()
                s = 'outlierRst{}'.format(capitalizedStatName)

                if hasattr(self, s):
                    return getattr(self, s)(col, **kwargs)

                else:
                    if s not in self._cache:
                        setattr(self._cache, s, {})
                    cache = getattr(self._cache, s)

                    if col not in cache:
                        verbose = True \
                            if arimo.debug.ON \
                            else kwargs.get('verbose')

                        if verbose:
                            tic = time.time()

                        outlierTails = kwargs.pop('outlierTails', 'both')

                        cache[col] = result = \
                            self.reprSample \
                                ._nonNullCol(
                                    col=col,
                                    lower=self.outlierRstMin(col)
                                        if outlierTails in ('lower', 'both')
                                        else None,
                                    upper=self.outlierRstMax(col)
                                        if outlierTails in ('upper', 'both')
                                        else None,
                                    strict=False) \
                                .select(getattr(sparkSQLFuncs, stat)(col)) \
                                .first()[0]

                        if result is None:
                            self.stdout_logger.warning(
                                msg='*** "{}" OUTLIER-RESISTANT {} = {} ***'.format(col, capitalizedStatName.upper(), result))
                            
                            result = self.outlierRstMin(col)

                        assert isinstance(result, (float, int))

                        if verbose:
                            toc = time.time()
                            self.stdout_logger.info(
                                msg='Outlier-Resistant {} for Column "{}" = {:,.3g}   <{:,.1f} s>'
                                    .format(capitalizedStatName, col, result, toc - tic))

                    return cache[col]

            else:
                raise ValueError(
                    '{0}.outlierRstStat({1}, ...): Column "{1}" Is Not of Numeric Type'
                        .format(self, col))

    def outlierRstMin(self, *cols, **kwargs):
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.outlierRstMin(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                if 'outlierRstMin' not in self._cache:
                    self._cache.outlierRstMin = {}

                if col not in self._cache.outlierRstMin:
                    verbose = True \
                        if arimo.debug.ON \
                        else kwargs.get('verbose')

                    if verbose:
                        tic = time.time()

                    outlierRstMin = \
                        self.reprSample \
                            .quantile(
                                col,
                                q=self._outlierTailProportion[col],
                                relativeError=0.)

                    sampleMin = self.sampleStat(col, stat='min')
                    sampleMedian = self.sampleMedian(col)

                    self._cache.outlierRstMin[col] = result = \
                        self.reprSample \
                            ._nonNullCol(
                                col=col,
                                lower=sampleMin,
                                strict=True) \
                            .select(sparkSQLFuncs.min(col)) \
                            .first()[0] \
                        if (outlierRstMin == sampleMin) and (outlierRstMin < sampleMedian) \
                        else outlierRstMin

                    assert isinstance(result, (float, int)), \
                        '*** "{}" OUTLIER-RESISTANT MIN = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Outlier-Resistant Min of Column "{}" = {:,.3g}   <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.outlierRstMin[col]

            else:
                raise ValueError(
                    '{0}.outlierRstMin({1}, ...): Column "{1}" Is Not of Numeric Type'
                        .format(self, col))

    def outlierRstMax(self, *cols, **kwargs):
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.outlierRstMax(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                if 'outlierRstMax' not in self._cache:
                    self._cache.outlierRstMax = {}

                if col not in self._cache.outlierRstMax:
                    verbose = True \
                        if arimo.debug.ON \
                        else kwargs.get('verbose')
                    
                    if verbose:
                        tic = time.time()

                    outlierRstMax = \
                        self.reprSample \
                            .quantile(
                                col,
                                q=1 - self._outlierTailProportion[col],
                                relativeError=0.)

                    sampleMax = self.sampleStat(col, stat='max')
                    sampleMedian = self.sampleMedian(col)

                    self._cache.outlierRstMax[col] = result = \
                        self.reprSample \
                            ._nonNullCol(
                                col=col,
                                upper=sampleMax,
                                strict=True) \
                            .select(sparkSQLFuncs.max(col)) \
                            .first()[0] \
                        if (outlierRstMax == sampleMax) and (outlierRstMax > sampleMedian) \
                        else outlierRstMax

                    assert isinstance(result, (float, int)), \
                        '*** "{}" OUTLIER-RESISTANT MAX = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Outlier-Resistant Max of Column "{}" = {:,.3g}   <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.outlierRstMax[col]

            else:
                raise ValueError(
                    '{0}.outlierRstMax({1}, ...): Column "{1}" Is Not of Numeric Type'
                        .format(self, col))

    def outlierRstMedian(self, *cols, **kwargs):
        if not cols:
            cols = self.possibleNumContentCols

        if len(cols) > 1:
            return Namespace(**
                {col: self.outlierRstMedian(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            if self.typeIsNum(col):
                if 'outlierRstMedian' not in self._cache:
                    self._cache.outlierRstMedian = {}

                if col not in self._cache.outlierRstMedian:
                    verbose = kwargs.get('verbose')
                    if verbose:
                        tic = time.time()

                    self._cache.outlierRstMedian[col] = result = \
                        self.reprSample \
                            ._nonNullCol(
                                col=col,
                                lower=self.outlierRstMin(col),
                                upper=self.outlierRstMax(col),
                                strict=False) \
                            .approxQuantile(
                                col=col,
                                probabilities=(.5,),
                                relativeError=0.)[0]

                    assert isinstance(result, (float, int)), \
                        '*** "{}" OUTLIER-RESISTANT MEDIAN = {} ***'.format(col, result)

                    if verbose:
                        toc = time.time()
                        self.stdout_logger.info(
                            msg='Outlier-Resistant Median of Column "{}" = {:,.3g}    <{:,.1f} s>'
                                .format(col, result, toc - tic))

                return self._cache.outlierRstMedian[col]

            else:
                raise ValueError(
                    '{0}.outlierRstMedian({1}, ...): Column "{1}" Is Not of Numeric Type'
                        .format(self, col))

    @_docstr_verbose
    def profile(self, *cols, **kwargs):
        """
        Return:
            *dict* of profile of salient statistics on specified columns of ``SparkADF``

        Args:
            *cols (str): names of column(s) to profile

            **kwargs:

                - **profileCat** *(bool, default = True)*: whether to profile possible categorical columns

                - **profileNum** *(bool, default = True)*: whether to profile numerical columns

                - **skipIfInvalid** *(bool, default = False)*: whether to skip profiling if column does not have enough non-NULLs
        """
        if not cols:
            cols = self.contentCols

        asDict = kwargs.pop('asDict', False)

        if len(cols) > 1:
            return Namespace(**
                {col: self.profile(col, **kwargs)
                 for col in cols})

        else:
            col = cols[0]

            verbose = True \
                if arimo.debug.ON \
                else kwargs.get('verbose')

            if verbose:
                msg = 'Profiling Column "{}"...'.format(col)
                self.stdout_logger.info(msg)
                tic = time.time()

            colType = self.type(col)
            profile = Namespace(type=colType)

            # non-NULL Proportions
            profile.nonNullProportion = \
                self.nonNullProportion(
                    col,
                    verbose=verbose > 1)

            if self.suffNonNull(col) or (not kwargs.get('skipIfInvalid', False)):
                # profile categorical column
                if kwargs.get('profileCat', True) and (colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _POSSIBLE_CAT_TYPES)):
                    profile.distinctProportions = \
                        self.distinct(
                            col,
                            count=True,
                            collect=True,
                            verbose=verbose > 1)

                # profile numerical column
                if kwargs.get('profileNum', True) and (colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _NUM_TYPES)):
                    outlierTailProportion = self._outlierTailProportion[col]

                    quantilesOfInterest = \
                        pandas.Series(
                            index=(0.,
                                   outlierTailProportion,
                                   .5,
                                   1 - outlierTailProportion,
                                   1.))
                    quantileProbsToQuery = []

                    sampleMin = self._cache.sampleMin.get(col)
                    if sampleMin:
                        quantilesOfInterest[0.] = sampleMin
                        toCacheSampleMin = False
                    else:
                        quantileProbsToQuery.append(0.)
                        toCacheSampleMin = True

                    outlierRstMin = self._cache.outlierRstMin.get(col)
                    if outlierRstMin:
                        quantilesOfInterest[outlierTailProportion] = outlierRstMin
                        toCacheOutlierRstMin = False
                    else:
                        quantileProbsToQuery.append(outlierTailProportion)
                        toCacheOutlierRstMin = True

                    sampleMedian = self._cache.sampleMedian.get(col)
                    if sampleMedian:
                        quantilesOfInterest[.5] = sampleMedian
                        toCacheSampleMedian = False
                    else:
                        quantileProbsToQuery.append(.5)
                        toCacheSampleMedian = True

                    outlierRstMax = self._cache.outlierRstMax.get(col)
                    if outlierRstMax:
                        quantilesOfInterest[1 - outlierTailProportion] = outlierRstMax
                        toCacheOutlierRstMax = False
                    else:
                        quantileProbsToQuery.append(1 - outlierTailProportion)
                        toCacheOutlierRstMax = True

                    sampleMax = self._cache.sampleMax.get(col)
                    if sampleMax:
                        quantilesOfInterest[1.] = sampleMax
                        toCacheSampleMax = False
                    else:
                        quantileProbsToQuery.append(1.)
                        toCacheSampleMax = True

                    if quantileProbsToQuery:
                        quantilesOfInterest[numpy.isnan(quantilesOfInterest)] = \
                            self.reprSample \
                                .quantile(
                                    col,
                                    q=tuple(quantileProbsToQuery),
                                    relativeError=0.)

                    sampleMin, outlierRstMin, sampleMedian, outlierRstMax, sampleMax = quantilesOfInterest

                    if toCacheSampleMin:
                        self._cache.sampleMin[col] = sampleMin

                    if toCacheOutlierRstMin:
                        if (outlierRstMin == sampleMin) and (outlierRstMin < sampleMedian):
                            outlierRstMin = \
                                self.reprSample \
                                    ._nonNullCol(
                                        col,
                                        lower=sampleMin,
                                        strict=True) \
                                    .select(sparkSQLFuncs.min(col)) \
                                    .first()[0]
                        self._cache.outlierRstMin[col] = outlierRstMin

                    if toCacheSampleMedian:
                        self._cache.sampleMedian[col] = sampleMedian

                    if toCacheOutlierRstMax:
                        if (outlierRstMax == sampleMax) and (outlierRstMax > sampleMedian):
                            outlierRstMax = \
                                self.reprSample \
                                    ._nonNullCol(
                                        col,
                                        upper=sampleMax,
                                        strict=True) \
                                    .select(sparkSQLFuncs.max(col)) \
                                    .first()[0]
                        self._cache.outlierRstMax[col] = outlierRstMax

                    if toCacheSampleMax:
                        self._cache.sampleMax[col] = sampleMax

                    profile.sampleRange = sampleMin, sampleMax
                    profile.outlierRstRange = outlierRstMin, outlierRstMax

                    profile.sampleMean = \
                        self.sampleStat(
                            col,
                            verbose=verbose)

                    profile.outlierRstMean = \
                        self._cache.outlierRstMean.get(
                            col,
                            self.outlierRstStat(
                                col,
                                verbose=verbose))

                    profile.outlierRstMedian = \
                        self._cache.outlierRstMedian.get(
                            col,
                            self.outlierRstMedian(
                                col,
                                verbose=verbose))

            if verbose:
                toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(toc - tic))

            return Namespace(**{col: profile}) \
                if asDict \
                else profile

    # *********
    # DATA PREP
    # fillna
    # prep

    @_docstr_verbose
    def fillna(self, *cols, **kwargs):
        """
        Fill/interpolate ``NULL``/``NaN`` values

        Return:
            ``SparkADF`` with ``NULL``/``NaN`` values filled/interpolated

        Args:
            *args (str): names of column(s) to fill/interpolate

            **kwargs:

                - **method** *(str)*: one of the following methods to fill ``NULL`` values in **numerical** columns,
                    or *dict* of such method specifications by column name

                    - ``avg``/``mean`` (default)
                    - ``min``
                    - ``max``
                    - ``avg_before``/``mean_before``
                    - ``min_before``
                    - ``max_before``
                    - ``avg_after``/``mean_after``
                    - ``min_after``
                    - ``max_after``
                    - ``linear`` (**TO-DO**)
                    - ``before`` (**TO-DO**)
                    - ``after`` (**TO-DO**)
                    - ``None`` (do nothing)

                    (*NOTE:* for an ``SparkADF`` with a ``.tCol`` set, ``NumPy/Pandas NaN`` values cannot be filled;
                        it is best that such *Python* values be cleaned up before they get into Spark)

                - **value**: single value, or *dict* of values by column name,
                    to use if ``method`` is ``None`` or not applicable
                    
                - **outlierTails** *(str or dict of str, default = 'both')*: specification of in which distribution tail (``None``, ``lower``, ``upper`` and ``both`` (default)) of each numerical column out-lying values may exist

                - **fillOutliers** *(bool or list of column names, default = False)*: whether to treat detected out-lying values as ``NULL`` values to be replaced in the same way

                - **loadPath** *(str)*: path to load existing ``NULL``-filling data transformations

                - **savePath** *(str)*: path to save new ``NULL``-filling data transformations
        """
        _TS_FILL_METHODS = \
            'avg_partition', 'mean_partition', 'min_partition', 'max_partition', \
            'avg_before', 'mean_before', 'min_before', 'max_before', \
            'avg_after', 'mean_after', 'min_after', 'max_after'

        if self.hasTS:
            _TS_OPPOSITE_METHODS = \
                Namespace(
                    avg='avg',
                    mean='mean',
                    min='max',
                    max='min')

            _TS_WINDOW_NAMES = \
                Namespace(
                    partition='partitionByI',
                    before='partitionByI_orderByT_before',
                    after='partitionByI_orderByT_after')

            _TS_OPPOSITE_WINDOW_NAMES = \
                Namespace(
                    partition='partition',
                    before='after',
                    after='before')

            _TS_WINDOW_DEFS = \
                Namespace(
                    partition=
                        '{} AS (PARTITION BY {}, {})'
                            .format(_TS_WINDOW_NAMES.partition, self._iCol, self._T_CHUNK_COL),
                    before=
                        '{} AS (PARTITION BY {}, {} ORDER BY {} ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)'
                            .format(_TS_WINDOW_NAMES.before, self._iCol, self._T_CHUNK_COL, self._T_ORD_COL),
                    after=
                        '{} AS (PARTITION BY {}, {} ORDER BY {} ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)'
                            .format(_TS_WINDOW_NAMES.after, self._iCol, self._T_CHUNK_COL, self._T_ORD_COL))

        returnDetails = kwargs.pop('returnDetails', False)
        returnSQLTransformer = kwargs.pop('returnSQLTransformer', False)
        loadPath = kwargs.pop('loadPath', None)
        savePath = kwargs.pop('savePath', None)

        verbose = kwargs.pop('verbose', False)
        if arimo.debug.ON:
            verbose = True

        if loadPath:
            if verbose:
                message = 'Loading NULL-Filling SQL Transformations from Paths "{}"...'.format(loadPath)
                self.stdout_logger.info(message)
                tic = time.time()

            sqlTransformer = \
                SQLTransformer.load(
                    path=loadPath)

            details = None

        else:
            value = kwargs.pop('value', None)

            method = kwargs.pop(
                'method',
                'mean' if value is None
                       else None)

            cols = set(cols)

            if isinstance(method, dict):
                cols.update(method)

            if isinstance(value, dict):
                cols.update(value)

            if not cols:
                cols = set(self.contentCols)

            cols.difference_update(
                self.indexCols +
                (self._T_ORD_COL, self._T_ORD_IN_CHUNK_COL))

            nulls = kwargs.pop('nulls', {})

            for col in cols:
                if col in nulls:
                    colNulls = nulls[col]
                    assert isinstance(colNulls, (list, tuple)) and (len(colNulls) == 2) \
                       and ((colNulls[0] is None) or isinstance(colNulls[0], (float, int))) \
                       and ((colNulls[1] is None) or isinstance(colNulls[1], (float, int)))

                else:
                    nulls[col] = (None, None)

            outlierTails = kwargs.pop('outlierTails', {})
            if isinstance(outlierTails, _STR_CLASSES):
                outlierTails = \
                    {col: outlierTails
                     for col in cols}

            fillOutliers = kwargs.pop('fillOutliers', False)
            fillOutliers = \
                cols \
                if fillOutliers is True \
                else to_iterable(fillOutliers)

            tsWindowDefs = set()
            details = {}

            if verbose:
                message = 'NULL-Filling Columns {}...'.format(
                    ', '.join('"{}"'.format(col) for col in cols))
                self.stdout_logger.info(message)
                tic = time.time()

            for col in cols:
                colType = self.type(col)
                colFallBackVal = None

                if colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _NUM_TYPES):
                    isNum = True

                    colOutlierTails = outlierTails.get(col, 'both')
                    fixLowerTail = colOutlierTails in ('lower', 'both')
                    fixUpperTail = colOutlierTails in ('upper', 'both')

                    methodForCol = \
                        method[col] \
                        if isinstance(method, dict) and (col in method) \
                        else method

                    if methodForCol:
                        methodForCol = methodForCol.lower().split('_')

                        if len(methodForCol) == 2:
                            assert self.hasTS, \
                                "NULL-Filling Methods {} Not Supported for Non-Time-Series SparkADFs".format(
                                    ', '.join(s.upper() for s in _TS_FILL_METHODS))

                            methodForCol, window = methodForCol

                        else:
                            methodForCol = methodForCol[0]

                            if self.hasTS:
                                window = None

                        colFallBackVal = \
                            self.outlierRstStat(
                                col,
                                stat=methodForCol
                                    if (not self.hasTS) or (window is None) or (window == 'partition')
                                    else 'mean',
                                outlierTails=colOutlierTails,
                                verbose=verbose > 1)

                    elif isinstance(value, dict):
                        colFallBackVal = value.get(col)
                        if not isinstance(colFallBackVal, _NUM_CLASSES):
                            colFallBackVal = None

                    elif isinstance(value, _NUM_CLASSES):
                        colFallBackVal = value

                else:
                    isNum = False

                    if isinstance(value, dict):
                        colFallBackVal = value.get(col)
                        if isinstance(colFallBackVal, _NUM_CLASSES):
                            colFallBackVal = None

                    elif not isinstance(value, _NUM_CLASSES):
                        colFallBackVal = value

                if pandas.notnull(colFallBackVal):
                    valFormatter = \
                        '%f' if colType.startswith(_DECIMAL_TYPE_PREFIX) or (colType in _FLOAT_TYPES) \
                             else ('%i' if colType in _INT_TYPES
                                        else ("'%s'" if (colType == _STR_TYPE) and
                                                        isinstance(colFallBackVal, _STR_CLASSES)
                                                     else '%s'))

                    fallbackStrs = [valFormatter % colFallBackVal]

                    lowerNull, upperNull = colNulls = nulls[col]

                    if isNum and self.hasTS and window:
                        partitionFallBackStrTemplate = \
                            "%s(CASE WHEN (STRING(%s) = 'NaN')%s%s%s%s THEN NULL ELSE %s END) OVER %s"

                        fallbackStrs.insert(0,
                            partitionFallBackStrTemplate
                                % (methodForCol,
                                   col,
                                   '' if lowerNull is None
                                      else ' OR ({} <= {})'.format(col, lowerNull),
                                   '' if upperNull is None
                                      else ' OR ({} >= {})'.format(col, upperNull),
                                   ' OR (%s < %s)' % (col, valFormatter % self.outlierRstMin(col))
                                        if fixLowerTail
                                        else '',
                                   ' OR (%s > %s)' % (col, valFormatter % self.outlierRstMax(col))
                                        if fixUpperTail
                                        else '',
                                   col,
                                   _TS_WINDOW_NAMES[window]))
                        tsWindowDefs.add(_TS_WINDOW_DEFS[window])

                        if window != 'partition':
                            oppositeWindow = _TS_OPPOSITE_WINDOW_NAMES[window]
                            fallbackStrs.insert(1,
                                partitionFallBackStrTemplate
                                    % (_TS_OPPOSITE_METHODS[methodForCol],
                                       col,
                                       '' if lowerNull is None
                                          else ' OR ({} <= {})'.format(col, lowerNull),
                                       '' if upperNull is None
                                          else ' OR ({} >= {})'.format(col, upperNull),
                                       ' OR (%s < %s)' % (col, valFormatter % self.outlierRstMin(col))
                                            if fixLowerTail
                                            else '',
                                       ' OR (%s > %s)' % (col, valFormatter % self.outlierRstMax(col))
                                            if fixUpperTail
                                            else '',
                                       col,
                                       _TS_WINDOW_NAMES[oppositeWindow]))
                            tsWindowDefs.add(_TS_WINDOW_DEFS[oppositeWindow])

                    details[col] = \
                        [self._NULL_FILL_PREFIX + col + self._PREP_SUFFIX,

                         dict(SQL="COALESCE(CASE WHEN (STRING({0}) = 'NaN'){1}{2}{3}{4} THEN NULL ELSE {0} END, {5})"
                                .format(
                                    col,
                                    '' if lowerNull is None
                                       else ' OR ({} <= {})'.format(col, lowerNull),
                                    '' if upperNull is None
                                       else ' OR ({} >= {})'.format(col, upperNull),
                                   ' OR ({} < {})'.format(col, valFormatter % self.outlierRstMin(col))
                                        if isNum and (col in fillOutliers) and fixLowerTail
                                        else '',
                                   ' OR ({} > {})'.format(col, valFormatter % self.outlierRstMax(col))
                                        if isNum and (col in fillOutliers) and fixUpperTail
                                        else '',
                                   ', '.join(fallbackStrs)),

                              Nulls=colNulls,
                              NullFillValue=colFallBackVal)]

            if tsWindowDefs:
                details['__TS_WINDOW_CLAUSE__'] = \
                    _tsWindowClause = \
                    'WINDOW {}'.format(', '.join(tsWindowDefs))

                if self._detPrePartitioned:
                    _tsWindowClause = \
                        _tsWindowClause.replace(
                            'PARTITION BY {}, {}'.format(self._iCol, self._T_CHUNK_COL),
                            'PARTITION BY {}'.format(self._iCol))

            else:
                _tsWindowClause = ''

            sqlTransformer = \
                SQLTransformer(
                    statement=
                        'SELECT *, {} FROM __THIS__ {}'
                            .format(
                                ', '.join(
                                    '{} AS {}'.format(nullFillDetails['SQL'], nullFillCol)
                                    for col, (nullFillCol, nullFillDetails) in details.items()
                                    if col != '__TS_WINDOW_CLAUSE__'),
                                _tsWindowClause))

        if savePath and (savePath != loadPath):
            if verbose:
                msg = 'Saving NULL-Filling SQL Transformations to Path "{}"...'.format(savePath)
                self.stdout_logger.info(msg)
                _tic = time.time()

            fs.rm(
                path=savePath,
                hdfs=arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS,
                is_dir=True,
                hadoop_home=arimo.backend._HADOOP_HOME)

            sqlTransformer.save(   # *** NEED TO ENHANCE TO ALLOW OVERWRITING ***
                path=savePath)

            if verbose:
                _toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - _tic))

        adf = self(
            sqlTransformer.transform,
            inheritNRows=True,
            **kwargs)

        adf._inheritCache(
            self,
            *(() if loadPath
                 else cols))

        adf._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdout_logger.info(message + ' done!   <{:,.1f} m>'.format((toc - tic) / 60))

        return ((adf, details, sqlTransformer)
                if returnSQLTransformer
                else (adf, details)) \
            if returnDetails \
            else adf

    @_docstr_verbose
    def prep(self, *cols, **kwargs):
        """
        Pre-process ``SparkADF``'s selected column(s) in standard ways:
            - One-hot-encode categorical columns
            - Scale numerical columns

        Return:
            Standard-pre-processed ``SparkADF``

        Args:
            *args: column(s) to pre-process

            **kwargs:
                - **forceCat** *(str or list/tuple of str, default = None)*: columns to force to be categorical variables

                - **forceNum** *(str or list/tuple of str, default = None)*: columns to force to be numerical variables

                - **fill**:
                    - *dict* ( ``method`` = ... *(default: 'mean')*, ``value`` = ... *(default: None)*, ``outlierTails`` = ... *(default: False)*, ``fillOutliers`` = ... *(default: False)*) as per ``.fillna(...)`` method;
                    - *OR* ``None`` to not apply any ``NULL``/``NaN``-filling

                - **scaler** *(str)*: one of the following methods to use on numerical columns
                    (*ignored* if loading existing ``prep`` pipeline from ``loadPath``):

                    - ``standard`` (default)
                    - ``maxabs``
                    - ``minmax``
                    - ``None`` *(do not apply any scaling)*

                - **assembleVec** *(str, default = '__X__')*: name of vector column to build from pre-processed features; *ignored* if loading existing ``prep`` pipeline from ``loadPath``

                - **loadPath** *(str)*: path to load existing data transformations

                - **savePath** *(str)*: path to save new fitted data transformations
        """
        def sqlStdScl(sqlItem, mean, std):
            return '(({}) - {}) / {}'.format(sqlItem, mean, std)

        def sqlMaxAbsScl(sqlItem, maxAbs):
            return '({}) / {}'.format(sqlItem, maxAbs)

        def sqlMinMaxScl(sqlItem, origMin, origMax, targetMin, targetMax):
            origRange = origMax - origMin
            targetRange = targetMax - targetMin
            return '({} * (({}) - ({})) / {}) + ({})'.format(
                targetRange, sqlItem, origMin, origRange, targetMin)

        nulls = kwargs.pop('nulls', {})

        forceCatIncl = kwargs.pop('forceCatIncl', None)
        forceCatExcl = kwargs.pop('forceCatExcl', None)
        forceCat = kwargs.pop('forceCat', None)
        forceCat = \
            (set()
                if forceCat is None
                else to_iterable(forceCat, iterable_type=set)) \
            .union(
                ()
                if forceCatIncl is None
                else to_iterable(forceCatIncl)) \
            .difference(
                ()
                if forceCatExcl is None
                else to_iterable(forceCatExcl))
        
        oheCat = kwargs.pop('oheCat', False)
        scaleCat = kwargs.pop('scaleCat', True)

        forceNumIncl = kwargs.pop('forceNumIncl', None)
        forceNumExcl = kwargs.pop('forceNumExcl', None)
        forceNum = kwargs.pop('forceNum', None)
        forceNum = \
            (set()
                if forceNum is None
                else to_iterable(forceNum, iterable_type=set)) \
            .union(
                ()
                if forceNumIncl is None
                else to_iterable(forceNumIncl)) \
            .difference(
                ()
                if forceNumExcl is None
                else to_iterable(forceNumExcl))

        fill = kwargs.pop(
            'fill',
            dict(method='mean',
                 value=None,
                 outlierTails='both',
                 fillOutliers=False))

        assert fill, '*** {}.prep(...) MUST INVOLVE NULL-FILLING FOR NUMERIC COLS ***'.format(type(self))

        scaler = kwargs.pop('scaler', 'standard')
        if scaler:
            scaler = scaler.lower()

        vecColsToAssemble = kwargs.pop('assembleVec', None)
        
        if not vecColsToAssemble:
            oheCat = False

        if oheCat:
            scaleCat = False

        returnOrigToPrepColMaps = kwargs.pop('returnOrigToPrepColMaps', False)
        returnPipeline = kwargs.pop('returnPipeline', False)

        loadPath = kwargs.pop('loadPath', None)
        savePath = kwargs.pop('savePath', None)

        verbose = kwargs.pop('verbose', False)
        if arimo.debug.ON:
            verbose = True

        if loadPath:
            if verbose:
                message = 'Loading & Applying Data Transformations from Path "{}"...'.format(loadPath)
                self.stdout_logger.info(message)
                tic = time.time()

            if loadPath in self._PREP_CACHE:
                prepCache = self._PREP_CACHE[loadPath]

                catOrigToPrepColMap = prepCache.catOrigToPrepColMap
                numOrigToPrepColMap = prepCache.numOrigToPrepColMap

                defaultVecCols = prepCache.defaultVecCols

                sqlStatement = prepCache.sqlStatement
                assert sqlStatement

                if prepCache.sqlTransformer is None:
                    prepCache.pipelineModelWithoutVectors = pipelineModelWithoutVectors = \
                        prepCache.sqlTransformer = sqlTransformer = SQLTransformer(statement=sqlStatement)

                    prepCache.catOHETransformer = catOHETransformer = None

                else:
                    sqlTransformer = prepCache.sqlTransformer
                    catOHETransformer = prepCache.catOHETransformer
                    pipelineModelWithoutVectors = prepCache.pipelineModelWithoutVectors

            else:
                if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                    localDirExists = os.path.isdir(loadPath)

                    hdfsDirExists = \
                        arimo.backend.hdfs.test(
                            path=loadPath,
                            exists=True,
                            directory=True)

                    if localDirExists and (not hdfsDirExists):
                        fs.put(
                            from_local=loadPath,
                            to_hdfs=loadPath,
                            is_dir=True,
                            _mv=False)

                    elif hdfsDirExists and (not localDirExists):
                        fs.get(
                            from_hdfs=loadPath,
                            to_local=loadPath,
                            is_dir=True,
                            overwrite=True, _mv=False,
                            must_succeed=True,
                            _on_linux_cluster_with_hdfs=True)

                catOrigToPrepColMap = \
                    json.load(open(os.path.join(loadPath, self._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'r'))

                numOrigToPrepColMap = \
                    json.load(open(os.path.join(loadPath, self._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'r'))

                defaultVecCols = \
                    [catOrigToPrepColMap[catCol][0]
                     for catCol in sorted(set(catOrigToPrepColMap)
                                          .difference(('__OHE__', '__SCALE__')))] + \
                    [numOrigToPrepColMap[numCol][0]
                     for numCol in sorted(set(numOrigToPrepColMap)
                                          .difference(('__TS_WINDOW_CLAUSE__', '__SCALER__')))]

                try:
                    pipelineModelWithoutVectors = sqlTransformer = SQLTransformer.load(path=loadPath)

                    sqlStatement = sqlTransformer.getStatement()

                    catOHETransformer = None

                except:
                    sqlStatementFilePath = \
                        os.path.join(loadPath, self._PREP_SQL_STATEMENT_FILE_NAME)

                    if os.path.isfile(sqlStatementFilePath):
                        sqlStatement = json.load(open(sqlStatementFilePath, 'r'))

                        pipelineModelWithoutVectors = sqlTransformer = SQLTransformer(statement=sqlStatement)

                        catOHETransformer = None

                    else:
                        pipelineModel = PipelineModel.load(path=loadPath)

                        nPipelineModelStages = len(pipelineModel.stages)

                        if nPipelineModelStages == 2:
                            sqlTransformer, secondTransformer = pipelineModel.stages

                            assert isinstance(sqlTransformer, SQLTransformer), \
                                '*** {} ***'.format(sqlTransformer)

                            sqlStatement = sqlTransformer.getStatement()

                            if isinstance(secondTransformer, OneHotEncoderModel):
                                catOHETransformer = secondTransformer
                                vectorAssembler = None
                                pipelineModelWithoutVectors = pipelineModel

                            elif isinstance(secondTransformer, VectorAssembler):
                                catOHETransformer = None
                                vectorAssembler = secondTransformer
                                pipelineModelWithoutVectors = sqlTransformer

                            else:
                                raise ValueError('*** {} ***'.format(secondTransformer))

                        elif nPipelineModelStages == 3:
                            sqlTransformer, catOHETransformer, vectorAssembler = pipelineModel.stages

                            assert isinstance(sqlTransformer, SQLTransformer), \
                                '*** {} ***'.format(sqlTransformer)

                            sqlStatement = sqlTransformer.getStatement()

                            assert isinstance(catOHETransformer, OneHotEncoderModel), \
                                '*** {} ***'.format(catOHETransformer)

                            assert isinstance(vectorAssembler, VectorAssembler), \
                                '*** {} ***'.format(vectorAssembler)

                            pipelineModelWithoutVectors = \
                                PipelineModel(stages=[sqlTransformer, catOHETransformer])

                        else:
                            raise ValueError('*** {} ***'.format(pipelineModel.stages))

                    if vectorAssembler:
                        vecInputCols = vectorAssembler.getInputCols()

                        assert set(defaultVecCols) == set(vecInputCols)

                        defaultVecCols = vecInputCols

                self._PREP_CACHE[loadPath] = \
                    Namespace(
                        catOrigToPrepColMap=catOrigToPrepColMap,
                        numOrigToPrepColMap=numOrigToPrepColMap,
                        defaultVecCols=defaultVecCols,

                        sqlStatement=sqlStatement,
                        sqlTransformer=sqlTransformer,

                        catOHETransformer=catOHETransformer,
                        pipelineModelWithoutVectors=pipelineModelWithoutVectors)
                
        else:
            if cols:
                cols = set(cols)

                cols = cols.intersection(self.possibleFeatureTAuxCols).union(
                        possibleFeatureContentCol for possibleFeatureContentCol in cols.intersection(self.possibleFeatureContentCols)
                                                  if self.suffNonNull(possibleFeatureContentCol))

            else:
                cols = self.possibleFeatureTAuxCols + \
                        tuple(possibleFeatureContentCol for possibleFeatureContentCol in self.possibleFeatureContentCols
                                                        if self.suffNonNull(possibleFeatureContentCol))

            if cols:
                profile = \
                    self.profile(
                        *cols,
                        profileCat=True,
                        profileNum=False,   # or bool(fill) or bool(scaler)?
                        skipIfInvalid=True,
                        asDict=True,
                        verbose=verbose)

            else:
                return self.copy()

            cols = {col for col in cols
                        if self.suffNonNull(col) and
                            (len(profile[col].distinctProportions
                                 .loc[# (profile[col].distinctProportions.index != '') &
                                      # FutureWarning: elementwise comparison failed; returning scalar instead,
                                      # but in the future will perform elementwise comparison
                                      pandas.notnull(profile[col].distinctProportions.index)]) > 1)}

            if not cols:
                return self.copy()

            catCols = \
                [col for col in cols.intersection(self.possibleCatCols).difference(forceNum)
                     if (col in forceCat) or
                        (profile[col].distinctProportions.iloc[:self._maxNCats[col]].sum()
                            >= self._minProportionByMaxNCats[col])]

            numCols = [col for col in cols.difference(catCols)
                           if self.typeIsNum(col)]

            cols = catCols + numCols

            if verbose:
                message = 'Prepping Columns {}...'.format(', '.join('"{}"'.format(col) for col in cols))
                self.stdout_logger.info(message)
                tic = time.time()

            prepSqlItems = {}

            catOrigToPrepColMap = \
                dict(__OHE__=oheCat,
                     __SCALE__=scaleCat)

            if catCols:
                if verbose:
                    msg = 'Transforming Categorical Features {}...'.format(
                        ', '.join('"{}"'.format(catCol) for catCol in catCols))
                    self.stdout_logger.info(msg)
                    _tic = time.time()

                catIdxCols = []
                if oheCat:
                    catOHECols = []
                elif scaleCat:
                    catScaledIdxCols = []

                for catCol in catCols:
                    catIdxCol = self._CAT_IDX_PREFIX + catCol + self._PREP_SUFFIX

                    catColType = self.type(catCol)

                    if catColType == _BOOL_TYPE:
                        cats = [0, 1]

                        nCats = 2

                        catIdxSqlItem = \
                            'CASE WHEN {0} IS NULL THEN 2 \
                                  WHEN {0} THEN 1 \
                                  ELSE 0 END'.format(catCol)

                    else:
                        isStr = (catColType == _STR_TYPE)

                        cats = [cat for cat in
                                        (profile[catCol].distinctProportions.index
                                         if catCol in forceCat
                                         else profile[catCol].distinctProportions.index[:self._maxNCats[catCol]])
                                    if (cat != '') and pandas.notnull(cat)]

                        nCats = len(cats)

                        catIdxSqlItem = \
                            'CASE {} ELSE {} END'.format(
                                ' '.join('WHEN {} = {} THEN {}'.format(
                                            catCol,
                                            "'{}'".format(cat.replace("'", "''").replace('"', '""'))
                                                if isStr
                                                else cat,
                                            i)
                                         for i, cat in enumerate(cats)),
                                nCats)

                    if oheCat:
                        catIdxCols.append(catIdxCol)

                        prepSqlItems[catIdxCol] = catIdxSqlItem

                        catPrepCol = self._OHE_PREFIX + catCol + self._PREP_SUFFIX
                        catOHECols.append(catPrepCol)

                    elif scaleCat:
                        catPrepCol = self._MIN_MAX_SCL_PREFIX + self._CAT_IDX_PREFIX + catCol + self._PREP_SUFFIX
                        catScaledIdxCols.append(catPrepCol)

                        prepSqlItems[catPrepCol] = \
                            sqlMinMaxScl(
                                sqlItem=catIdxSqlItem,
                                origMin=0, origMax=nCats,
                                targetMin=-1, targetMax=1)

                    else:
                        prepSqlItems[catIdxCol] = catIdxSqlItem

                        catPrepCol = catIdxCol

                    catOrigToPrepColMap[catCol] = \
                        [catPrepCol,

                         dict(Cats=cats,
                              NCats=nCats)]

                if oheCat:
                    catOHETransformer = \
                        OneHotEncoder(
                            inputCols=catIdxCols,
                            outputCols=catOHECols,
                            handleInvalid='error',
                                # 'keep': invalid data presented as an extra categorical feature
                                # When handleInvalid is configured to 'keep',
                                # an extra "category" indicating invalid values is added as last category,
                                # so when dropLast is true, invalid values are encoded as all-zeros vector
                            dropLast=True) \
                        .fit(dataset=self.reprSample._sparkDF[catCols].union(
                                        arimo.backend.spark.sql(
                                            'VALUES ({})'.format(', '.join(len(catCols) * ('NULL',)))))
                                    .selectExpr(*('{} AS {}'.format(catSqlItem, strIdxCol)
                                                  for strIdxCol, catSqlItem in prepSqlItems.items())))

                else:
                    catOHETransformer = None

                if verbose:
                    _toc = time.time()
                    self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - tic))

            else:
                catOHETransformer = None

            numOrigToPrepColMap = \
                dict(__SCALER__=scaler)

            if numCols:
                numScaledCols = []

                if verbose:
                    msg = 'Transforming Numerical Features {}...'.format(
                        ', '.join('"{}"'.format(numCol) for numCol in numCols))
                    self.stdout_logger.info(msg)
                    _tic = time.time()

                outlierTails = fill.get('outlierTails', {})
                if isinstance(outlierTails, _STR_CLASSES):
                    outlierTails = \
                        {col: outlierTails
                         for col in numCols}

                _, numNullFillDetails = \
                    self.fillna(
                        *numCols,
                        nulls=nulls,
                        method=fill.get('method', 'mean'),
                        value=fill.get('value'),
                        outlierTails=outlierTails,
                        fillOutliers=fill.get('fillOutliers', False),
                        returnDetails=True,
                        verbose=verbose > 1)

                for numCol in numCols:
                    colOutlierTails = outlierTails.get(numCol, 'both')

                    excludeLowerTail = colOutlierTails in ('lower', 'both')
                    colMin = self.outlierRstMin(numCol) \
                        if excludeLowerTail \
                        else self.sampleStat(numCol, stat='min')

                    excludeUpperTail = colOutlierTails in ('upper', 'both')
                    colMax = self.outlierRstMax(numCol) \
                        if excludeUpperTail \
                        else self.sampleStat(numCol, stat='max')

                    if colMin < colMax:
                        numColNullFillDetails = numNullFillDetails[numCol][1]

                        numColSqlItem = numColNullFillDetails['SQL']
                        numColNulls = numColNullFillDetails['Nulls']
                        nulColNullFillValue = numColNullFillDetails['NullFillValue']

                        if scaler:
                            if scaler == 'standard':
                                scaledCol = self._STD_SCL_PREFIX + numCol + self._PREP_SUFFIX

                                mean = self.outlierRstStat(numCol)

                                stdDev = \
                                    self.reprSample \
                                        ._nonNullCol(
                                            col=numCol,
                                            lower=colMin if excludeLowerTail else None,
                                            upper=colMax if excludeUpperTail else None,
                                            strict=False) \
                                        .select(sparkSQLFuncs.stddev(numCol)) \
                                        .first()[0]

                                prepSqlItems[scaledCol] = \
                                    sqlStdScl(
                                        sqlItem=numColSqlItem,
                                        mean=mean,
                                        std=stdDev)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=nulColNullFillValue,
                                          Mean=mean,
                                          StdDev=stdDev)]

                            elif scaler == 'maxabs':
                                scaledCol = self._MAX_ABS_SCL_PREFIX + numCol + self._PREP_SUFFIX

                                maxAbs = max(abs(colMin), abs(colMax))

                                prepSqlItems[scaledCol] = \
                                    sqlMaxAbsScl(
                                        sqlItem=numColSqlItem,
                                        maxAbs=maxAbs)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=nulColNullFillValue,
                                          MaxAbs=maxAbs)]

                            elif scaler == 'minmax':
                                scaledCol = self._MIN_MAX_SCL_PREFIX + numCol + self._PREP_SUFFIX

                                prepSqlItems[scaledCol] = \
                                    sqlMinMaxScl(
                                        sqlItem=numColSqlItem,
                                        origMin=colMin, origMax=colMax,
                                        targetMin=-1, targetMax=1)

                                numOrigToPrepColMap[numCol] = \
                                    [scaledCol,

                                     dict(Nulls=numColNulls,
                                          NullFillValue=nulColNullFillValue,
                                          OrigMin=colMin, OrigMax=colMax,
                                          TargetMin=-1, TargetMax=1)]

                            else:
                                raise ValueError('*** Scaler must be one of "standard", "maxabs", "minmax" and None ***')

                        else:
                            scaledCol = self._NULL_FILL_PREFIX + numCol + self._PREP_SUFFIX

                            prepSqlItems[scaledCol] = numColSqlItem

                            numOrigToPrepColMap[numCol] = \
                                [scaledCol,

                                 dict(Nulls=numColNulls,
                                      NullFillValue=nulColNullFillValue)]

                        numScaledCols.append(scaledCol)

                if verbose:
                    _toc = time.time()
                    self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - _tic))

            defaultVecCols = \
                [catOrigToPrepColMap[catCol][0]
                 for catCol in sorted(set(catOrigToPrepColMap)
                                      .difference(('__OHE__', '__SCALE__')))] + \
                [numOrigToPrepColMap[numCol][0]
                 for numCol in sorted(set(numOrigToPrepColMap)
                                      .difference(('__TS_WINDOW_CLAUSE__', '__SCALER__')))]

            sqlStatement = \
                'SELECT *, {} FROM __THIS__ {}'.format(
                    ', '.join('{} AS {}'.format(sqlItem, prepCol)
                              for prepCol, sqlItem in prepSqlItems.items()),
                    numNullFillDetails.get('__TS_WINDOW_CLAUSE__', ''))

            sqlTransformer = SQLTransformer(statement=sqlStatement)

            pipelineModelWithoutVectors = \
                PipelineModel(stages=[sqlTransformer, catOHETransformer]) \
                if catCols and oheCat \
                else sqlTransformer

        if savePath and (savePath != loadPath):
            if verbose:
                msg = 'Saving Data Transformations to Local Path {}...'.format(savePath)
                self.stdout_logger.info(msg)
                _tic = time.time()

            # *** NEED TO ENHANCE TO ALLOW OVERWRITING ***
            fs.rm(
                path=savePath,
                hdfs=arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS,
                is_dir=True,
                hadoop_home=arimo.backend._HADOOP_HOME)

            pipelineModelWithoutVectors.save(path=savePath)

            if arimo.backend._ON_LINUX_CLUSTER_WITH_HDFS:
                fs.get(
                    from_hdfs=savePath,
                    to_local=savePath,
                    is_dir=True,
                    _mv=False)

            json.dump(
                catOrigToPrepColMap,
                open(os.path.join(savePath, self._CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'w'),
                indent=2)

            json.dump(
                numOrigToPrepColMap,
                open(os.path.join(savePath, self._NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME), 'w'),
                indent=2)

            json.dump(
                sqlStatement,
                open(os.path.join(savePath, self._PREP_SQL_STATEMENT_FILE_NAME), 'w'),
                indent=2)

            if verbose:
                _toc = time.time()
                self.stdout_logger.info(msg + ' done!   <{:,.1f} s>'.format(_toc - _tic))

            self._PREP_CACHE[savePath] = \
                Namespace(
                    catOrigToPrepColMap=catOrigToPrepColMap,
                    numOrigToPrepColMap=numOrigToPrepColMap,

                    defaultVecCols=defaultVecCols,

                    sqlStatement=sqlStatement,
                    sqlTransformer=sqlTransformer,

                    catOHETransformer=catOHETransformer,
                    pipelineModelWithoutVectors=pipelineModelWithoutVectors)

        if self._detPrePartitioned and self.hasTS:
            _partitionBy_str = 'PARTITION BY {}, {}'.format(self._iCol, self._T_CHUNK_COL)

            statement = sqlTransformer.getStatement()

            if _partitionBy_str in statement:
                sqlTransformer = \
                    SQLTransformer(
                        statement=
                            statement.replace(
                                _partitionBy_str,
                                'PARTITION BY {}'.format(self._iCol)))

        pipelineModelStages = \
            [sqlTransformer] + \
            ([catOHETransformer]
             if catOHETransformer
             else [])

        if vecColsToAssemble:
            if isinstance(vecColsToAssemble, _STR_CLASSES):
                pipelineModelStages.append(
                    VectorAssembler(
                        inputCols=defaultVecCols,
                        outputCol=vecColsToAssemble))

            else:
                assert isinstance(vecColsToAssemble, (dict, Namespace))

                for vecOutputCol, vecInputCols in vecColsToAssemble.items():
                    pipelineModelStages.append(
                        VectorAssembler(
                            inputCols=vecInputCols
                                if vecInputCols
                                else defaultVecCols,
                            outputCol=vecOutputCol))

        pipelineModel = \
            PipelineModel(stages=pipelineModelStages) \
            if len(pipelineModelStages) > 1 \
            else sqlTransformer
        
        try:   # in case SELF is FilesBasedADF
            sparkDF = self._initSparkDF
        except:
            sparkDF = self._sparkDF

        missingCatCols = \
            set(catOrigToPrepColMap) \
            .difference(
                sparkDF.columns +
                ['__OHE__', '__SCALE__'])

        missingNumCols = \
            set(numOrigToPrepColMap) \
            .difference(
                sparkDF.columns +
                ['__TS_WINDOW_CLAUSE__', '__SCALER__'])

        if missingCatCols or missingNumCols:
            if arimo.debug.ON:
                self.stdout_logger.debug(
                    msg='*** FILLING MISSING COLS {} ***'
                        .format(missingCatCols | missingNumCols))

            sparkDF = \
                sparkDF.select(
                    '*',
                    *([sparkSQLFuncs.lit(None)
                            .alias(missingCatCol)
                       for missingCatCol in missingCatCols] +
                      [sparkSQLFuncs.lit(numOrigToPrepColMap[missingNumCol][1]['NullFillValue'])
                            .alias(missingNumCol)
                       for missingNumCol in missingNumCols]))

        colsToKeep = \
            sparkDF.columns + \
            (to_iterable(vecColsToAssemble, iterable_type=list)
             if vecColsToAssemble
             else (([catPrepColDetails[0]
                     for catCol, catPrepColDetails in catOrigToPrepColMap.items()
                     if (catCol not in ('__OHE__', '__SCALE__')) and
                        isinstance(catPrepColDetails, list) and (len(catPrepColDetails) == 2)] +
                    [numPrepColDetails[0]
                     for numCol, numPrepColDetails in numOrigToPrepColMap.items()
                     if (numCol not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and
                        isinstance(numPrepColDetails, list) and (len(numPrepColDetails) == 2)])
                   if loadPath
                   else (((catScaledIdxCols
                           if scaleCat
                           else catIdxCols)
                          if catCols
                          else []) +
                         (numScaledCols
                          if numCols
                          else []))))

        adf = self._decorate(
            obj=pipelineModel.transform(dataset=sparkDF)[colsToKeep],
            nRows=self._cache.nRows,
            **kwargs)

        adf._inheritCache(
            self,
            *(() if loadPath
                 else colsToKeep))

        adf._cache.reprSample = self._cache.reprSample

        if verbose:
            toc = time.time()
            self.stdout_logger.info(message + ' done!   <{:,.1f} m>'.format((toc - tic) / 60))

        return ((adf, catOrigToPrepColMap, numOrigToPrepColMap, pipelineModel)
                if returnPipeline
                else (adf, catOrigToPrepColMap, numOrigToPrepColMap)) \
            if returnOrigToPrepColMaps \
            else adf

    # *******************************
    # ITERATIVE GENERATION / SAMPLING
    # _collectCols
    # collect
    # _prepareArgsForSampleOrGenOrPred
    # sample
    # gen

    def _collectCols(self, pandasDF_or_rddRows, cols, asPandas=True,
                     overTime=False, padUpToNTimeSteps=0, padValue=numpy.nan, padBefore=True):
        def pad(array, nRows, padValue=numpy.nan):
            shape = array.shape
            existingNRows = shape[0]
    
            if (padValue is None) or (nRows <= existingNRows):
                return array

            else:
                assert padBefore is not None, '*** "padBefore" argument must not be None ***'

                padArray = numpy.full(
                    shape=(nRows - existingNRows,) + shape[1:],
                    fill_value=padValue)

                return numpy.vstack(
                    (padArray, array)
                    if padBefore
                    else (array, padArray))

        nRows = len(pandasDF_or_rddRows)
        cols = to_iterable(cols, iterable_type=list)

        if asPandas:
            return pandasDF_or_rddRows[cols] \
                if isinstance(pandasDF_or_rddRows, pandas.DataFrame) \
                else pandas.DataFrame.from_records(
                        data=pandasDF_or_rddRows,
                        index=None,
                        exclude=None,
                        columns=pandasDF_or_rddRows[0].__fields__,
                        coerce_float=False,
                        nrows=None)

        elif overTime:
            assert len(cols) == 1, '*** OVER-TIME COLS TO COLLECT: {} ***'.format(cols)
            col = cols[0]

            if padValue is None:
                padValue = numpy.nan

            lists = pandasDF_or_rddRows[col].tolist() \
                    if isinstance(pandasDF_or_rddRows, pandas.DataFrame) \
                    else [row[col]
                          for row in pandasDF_or_rddRows]

            colWidth = self._colWidth(col)

            try:
                firstNonEmptyRow = \
                    next(lists[i]
                         for i in range(nRows)
                         if lists[i])

                firstValue = firstNonEmptyRow[0]

                arrayForEmptyRows = \
                    numpy.full(
                        shape=(1, padUpToNTimeSteps, colWidth),
                        fill_value=padValue)

                if isinstance(firstValue, Vector):
                    return numpy.vstack(
                            (numpy.expand_dims(
                                pad(array=numpy.vstack(v.toArray() for v in ls),
                                    nRows=padUpToNTimeSteps,
                                    padValue=padValue),
                                axis=0)
                             if ls
                             else arrayForEmptyRows)
                            for ls in lists)

                elif isinstance(firstValue, (list, tuple)):
                    isVector = [isinstance(v, Vector) for v in firstValue]
                    isIterable = [isinstance(v, (list, tuple)) for v in firstValue]

                    return numpy.vstack(
                            (numpy.expand_dims(
                                pad(array=numpy.array([flatten(v) for v in ls]),
                                    nRows=padUpToNTimeSteps,
                                    padValue=padValue),
                                axis=0)
                             if ls
                             else arrayForEmptyRows)
                            for ls in lists) \
                        if any(isVector) or any(isIterable) \
                      else numpy.vstack(
                            (numpy.expand_dims(
                                pad(array=numpy.vstack(ls),
                                    nRows=padUpToNTimeSteps,
                                    padValue=padValue),
                                axis=0)
                             if ls
                             else arrayForEmptyRows)
                            for ls in lists)

                else:
                    return numpy.vstack(
                            numpy.expand_dims(
                                pad(array=numpy.expand_dims(ls, axis=1),
                                    nRows=padUpToNTimeSteps,
                                    padValue=padValue),
                                axis=0)
                            for ls in lists)

            except StopIteration:
                return numpy.full(
                        shape=(nRows, padUpToNTimeSteps, colWidth),
                        fill_value=padValue)

        else:
            isVector = []
            isIterable = []

            if isinstance(pandasDF_or_rddRows, pandas.DataFrame):
                for col in cols:
                    firstValue = pandasDF_or_rddRows[col].iat[0]
                    isVector.append(isinstance(firstValue, Vector))
                    isIterable.append(isinstance(firstValue, (list, tuple)))

                return numpy.hstack(
                        (numpy.vstack(v.toArray()
                                      for v in pandasDF_or_rddRows[col])
                         if isVector[i]
                         else (numpy.vstack(flatten(v)
                                            for v in pandasDF_or_rddRows[col])
                               if isIterable[i]
                               else pandasDF_or_rddRows[[col]].values))
                        for i, col in enumerate(cols)) \
                    if any(isVector) or any(isIterable) \
                    else pandasDF_or_rddRows[cols].values

            else:
                for col in cols:
                    firstValue = pandasDF_or_rddRows[0][col]
                    isVector.append(isinstance(firstValue, Vector))
                    isIterable.append(isinstance(firstValue, (list, tuple)))

                return numpy.vstack(
                        numpy.hstack(
                            (row[col].toArray()
                             if isVector[i]
                             else (flatten(row[col])
                                   if isIterable[i]
                                   else row[col]))
                            for i, col in enumerate(cols))
                        for row in pandasDF_or_rddRows)

    def collect(self, *colsLists, **kwargs):
        anon = kwargs.get('anon', False)
        asPandas = kwargs.get('pandas', True)

        if colsLists:
            colsLists = \
                ([]
                 if anon
                 else [self.indexCols]) + \
                list(colsLists)

            df = self._sparkDF \
                .select(*set(flatten(colsLists))) \
                .toPandas()

            return [self._collectCols(df, cols, asPandas=asPandas)
                    for cols in colsLists] \
                if len(colsLists) > 1 \
                else self._collectCols(df, cols=colsLists[0], asPandas=asPandas)

        else:
            return self._collectCols(
                self.toPandas(),
                cols=(self.possibleFeatureTAuxCols + self.contentCols)
                    if anon
                    else self.columns,
                asPandas=asPandas)

    @lru_cache()
    def _prepareArgsForSampleOrGenOrPred(self, *args, **kwargs):
        def cols_rowFrom_rowTo(x):
            if isinstance(x, _STR_CLASSES):
                return [x], None, None
            elif isinstance(x, (list, tuple)) and x:
                lastItem = x[-1]
                if isinstance(lastItem, _STR_CLASSES):
                    return x, None, None
                elif isinstance(lastItem, int):
                    secondLastItem = x[-2]
                    return ((x[:-1], 0, lastItem)
                            if lastItem >= 0
                            else (x[:-1], lastItem, 0)) \
                        if isinstance(secondLastItem, _STR_CLASSES) \
                        else (x[:-2], secondLastItem, lastItem)

        def windowRowStr(n, sep=' '):
            return '{}{}Following'.format(n, sep) \
                if n > 0 \
                else ('{}{}Preceding'.format(-n, sep)
                      if n < 0
                      else 'Current{}Row'.format(sep))

        withReplacement = kwargs.get('withReplacement', False)
        seed = kwargs.get('seed')

        collect = kwargs.get('collect')
        anon = kwargs.get('anon', False)

        filterCondition = kwargs.get('filter')

        if filterCondition:
            if arimo.debug.ON:
                self.stdout_logger.debug(
                    '*** FILTER CONDITION: {} ***'.format(filterCondition))

            # only relevant for Time Series scoring use cases
            keepOrigRows = kwargs.get('keepOrigRows')

        overTimeColWidths = {}
        
        finalTotalNRows = None

        if args:
            alias = None

            if self.hasTS:
                windowDefs = set()

                sqlItems = set(self.indexCols + self.tAuxCols)

                if filterCondition:
                    _FILTER_COL_NAME = '__FILTER__'

                    # explicitly compute the filter Booleans
                    # just in case Catalyst optimizer cannot simplify the filter condition
                    # across different SELECT'ed items
                    # and also because mixing WHERE & WINDOW in 1 SQL statement often causes errors
                    adf = self(
                        '*',
                        '({}) AS {}'.format(
                            filterCondition,
                            _FILTER_COL_NAME),
                        inheritCache=True,
                        inheritNRows=True)

                colsLists = []
                colsOverTime = []
                padUpToNTimeSteps = []
                padBefore = []
                padValue = kwargs.get('pad')

                for cols, rowFrom, rowTo in map(cols_rowFrom_rowTo, args):
                    if (rowFrom is None) and (rowTo is None):
                        sqlItems.update(cols)
                        colsLists.append(cols)
                        colsOverTime.append(False)
                        padUpToNTimeSteps.append(None)
                        padBefore.append(None)

                    else:
                        windowName = \
                            'partitionByI_orderByT_{}_{}'.format(
                                windowRowStr(rowFrom, sep=''),
                                windowRowStr(rowTo, sep=''))

                        windowDefs.add(
                            '{} AS (PARTITION BY {}{} ORDER BY {} ROWS BETWEEN {} AND {})'
                                .format(
                                    windowName,

                                    self._iCol,
                                    '' if self._detPrePartitioned
                                        else ', {}'.format(self._T_CHUNK_COL),

                                    self._T_ORD_COL,

                                    windowRowStr(rowFrom),
                                    windowRowStr(rowTo)))

                        windowFromRowStr = windowRowStr(rowFrom, sep='')
                        windowToRowStr = windowRowStr(rowTo, sep='')

                        # COLLECT_LIST of NAMED_STRUCTs
                        sqlItem = \
                            'COLLECT_LIST(NAMED_STRUCT({})) OVER {}'.format(
                                ', '.join("'{0}', {0}".format(col)
                                          for col in cols),
                                windowName)

                        # NAMED_STRUCT OF COLLECT_LISTs
                        # sqlItem = \
                        #     'NAMED_STRUCT({})'.format(
                        #         ', '.join(
                        #             "'__{0}__from{1}_to{2}__', COLLECT_LIST({0}) OVER {3}"
                        #                 .format(col, windowFromRowStr, windowToRowStr, windowName)
                        #             for col in cols))

                        resultCol = \
                            '__{}__from{}_to{}__'.format(
                                '_n_'.join(cols),
                                windowFromRowStr,
                                windowToRowStr)

                        sqlItems.add(
                            'IF({}, {}, NULL) AS {}'.format(
                                _FILTER_COL_NAME,
                                sqlItem,
                                resultCol)
                            if filterCondition
                            else '{} AS {}'.format(
                                sqlItem,
                                resultCol))

                        colsLists.append([resultCol])
                        colsOverTime.append(True)
                        padUpToNTimeSteps.append(rowTo - rowFrom + 1)
                        padBefore.append(rowFrom < 0)
                        overTimeColWidths[resultCol] = \
                            sum(self._colWidth(*cols, asDict=True).values())

                if filterCondition:
                    if keepOrigRows:
                        adf('SELECT {} FROM this {}'.format(
                                ', '.join(sqlItems),
                                'WINDOW {}'.format(', '.join(windowDefs))
                                    if windowDefs
                                    else ''),
                            inheritCache=True,
                            inheritNRows=True,
                            inplace=True)

                    else:
                        adf = adf(
                            'SELECT {}, {} FROM this {}'.format(
                                ', '.join(sqlItems),
                                _FILTER_COL_NAME,
                                'WINDOW {}'.format(', '.join(windowDefs))
                                    if windowDefs
                                    else ''),
                            inheritCache=True,
                            inheritNRows=True,
                            # tCol=None   # important for using _T_ORD_COL for subsequent joins
                            ).filter(condition=_FILTER_COL_NAME) \
                            .drop(_FILTER_COL_NAME)

                        finalTotalNRows = adf.nRows

                else:
                    adf = self(
                        'SELECT {} FROM this {}'.format(
                            ', '.join(sqlItems),
                            'WINDOW {}'.format(', '.join(windowDefs))
                                if windowDefs
                                else ''),
                        inheritCache=True,
                        inheritNRows=True,
                        # tCol=None   # important for using _T_ORD_COL for subsequent joins
                        )   

            else:
                colsToKeep = set(self.indexCols).union(flatten(args))

                if filterCondition:
                    adf = self.filter(filterCondition)[colsToKeep]
                    finalTotalNRows = adf.nRows

                else:
                    adf = self[colsToKeep]

                colsLists = list(args)
                nArgs = len(args)
                colsOverTime = nArgs * [False]
                padUpToNTimeSteps = nArgs * [None]
                padBefore = nArgs * [None]
                padValue = None

        else:
            alias = kwargs.get('alias')

            if filterCondition:
                adf = self.filter(filterCondition)
                finalTotalNRows = adf.nRows

            else:
                adf = self.copy()

            colsLists = [self.possibleFeatureTAuxCols + self.contentCols]
            colsOverTime = [False]
            padUpToNTimeSteps = [None]
            padBefore = [None]
            padValue = None

        adf._cache.colWidth.update(overTimeColWidths)

        n = kwargs.get('n', self._DEFAULT_REPR_SAMPLE_SIZE)
        if n:
            fraction = min(n / adf.nRows, 1.)
        else:
            fraction = kwargs.get('fraction')
            if fraction and finalTotalNRows:
                fraction = min((self.nRows / finalTotalNRows) * fraction, 1.)

        return Namespace(
            adf=adf,
            colsLists=colsLists,
            colsOverTime=colsOverTime,
            padUpToNTimeSteps=padUpToNTimeSteps,
            padBefore=padBefore,
            padValue=padValue,
            withReplacement=withReplacement,
            n=n,
            fraction=fraction,
            seed=seed,
            collect=collect,
            anon=anon,
            alias=alias)

    def sample(self, *args, **kwargs):
        """
        Return:
            - if ``collect = False / None``: return 1 or a *list* of ``SparkADF`` s
            - if ``collect = 'pandas'``: return 1 or a *list* of ``Pandas DataFrame`` s
            - if ``collect = 'numpy'``: return 1 or a *list* of ``NumPy Array`` s

        Args:
            *args:
                - series of (``'col_name_1'``, ``'col_name_2'``, ... ``[, from_relative_time_slice, to_relative_time_slice]``) tuples
                - if empty, sample all existing columns

            **kwargs:
                - **n** *(int, default = None)*: *approximate* number of rows to sample

                - **fraction** *(float between 0 and 1, default = 1e-3)*: fraction of rows to sample; *ignored* if ``n`` is given

                - **withReplacement** *(bool, default = False)*: whether to sample with replacement

                - **seed** *(int)*: randomizer seed

                - **anon** *(bool, default = False)*: whether to include index columns ``.iCol`` and ``.tCol`` in the result(s)

                - **collect**:
                    - ``False`` / ``None``: do not collect the result(s) to ``Pandas DataFrame``(s) / ``NumPy Array`` s
                    - ``pandas``: collect result(s) to ``Pandas DataFrame`` s
                    - ``numpy``: collect result(s) to ``NumPy Array`` s

                - **padValue**: *(only applicable to time-series data)* value to pad short extracted series to desired max series length

                - **alias** *(str, default = None)*: name of the resulting sampled ``SparkADF``; only applicable when ``*args`` is empty and ``collect = False``
        """
        preparedArgs = self._prepareArgsForSampleOrGenOrPred(*args, **kwargs)

        if preparedArgs.collect:
            df = ()

            while not len(df):
                df = preparedArgs.adf._sparkDF.sample(
                        withReplacement=preparedArgs.withReplacement,
                        fraction=preparedArgs.fraction,
                        seed=preparedArgs.seed) \
                    .toPandas()

            if not preparedArgs.anon:
                preparedArgs.colsLists.insert(0, self.indexCols)
                preparedArgs.colsOverTime.insert(0, False)
                preparedArgs.padUpToNTimeSteps.insert(0, None)
                preparedArgs.padBefore.insert(0, None)

            asPandas = preparedArgs.collect.lower() == 'pandas'

            if len(preparedArgs.colsLists) > 1:
                return [df[cols] for cols in preparedArgs.colsLists] \
                    if asPandas \
                    else [preparedArgs.adf._collectCols(
                            df, cols,
                            asPandas=False,
                            overTime=preparedArgs.colsOverTime[i],
                            padUpToNTimeSteps=preparedArgs.padUpToNTimeSteps[i],
                            padValue=preparedArgs.padValue,
                            padBefore=preparedArgs.padBefore[i])
                          for i, cols in enumerate(preparedArgs.colsLists)]

            else:
                cols = preparedArgs.colsLists[0]
                df = df[cols]
                return df \
                    if asPandas \
                    else preparedArgs.adf._collectCols(
                            df, cols,
                            asPandas=False,
                            overTime=preparedArgs.colsOverTime[0],
                            padUpToNTimeSteps=preparedArgs.padUpToNTimeSteps[0],
                            padValue=preparedArgs.padValue,
                            padBefore=preparedArgs.padBefore[0])

        else:
            if preparedArgs.fraction < 1:
                sparkDF = \
                    preparedArgs.adf._sparkDF.sample(
                        withReplacement=preparedArgs.withReplacement,
                        fraction=preparedArgs.fraction,
                        seed=preparedArgs.seed)

                nRows = None

            else:
                sparkDF = preparedArgs.adf._sparkDF

                nRows = preparedArgs.adf._cache.nRows

            adf = self._decorate(
                obj=sparkDF[[col for col in set(flatten(preparedArgs.colsLists))]],
                nRows=nRows,
                iCol=self._iCol, tCol=None,   # *** SAMPLES are UNORDERED ***
                alias=preparedArgs.alias)

            # inherit col widths, some of which may have been calculated in special ways
            adf._cache.colWidth.update(preparedArgs.adf._cache.colWidth)

            return adf

    def gen(self, *args, **kwargs):
        """
        Same as ``.sample(...)``, but produces a **generator**

        Keyword Args:
            - **cache** *(bool, default = True)*: whether to cache the SparkADF before generating samples
        """
        preparedArgs = self._prepareArgsForSampleOrGenOrPred(*args, **kwargs)

        if not preparedArgs.anon:
            preparedArgs.colsLists.insert(0, self.indexCols)
            preparedArgs.colsOverTime.insert(0, False)
            preparedArgs.padUpToNTimeSteps.insert(0, None)
            preparedArgs.padBefore.insert(0, None)

        asPandas = False \
            if preparedArgs.collect is None \
            else (preparedArgs.collect.lower() == 'pandas')

        sampleN = kwargs.get('sampleN')
        
        if sampleN:
            if sampleN < preparedArgs.adf.nRows:
                adf = preparedArgs.adf.sample(
                    n=sampleN,
                    anon=preparedArgs.anon)

                adf._cache.colWidth = copy.deepcopy(preparedArgs.adf._cache.colWidth)

            else:
                sampleN = None

                adf = preparedArgs.adf

                if arimo.debug.ON:
                    self.stdout_logger.debug(msg='*** .gen(...): NOT TRIGGERING SAMPLING: sampleN >= Data Size ***')

        else:
            adf = preparedArgs.adf

            if arimo.debug.ON:
                self.stdout_logger.debug(msg='*** .gen(...): NOT TRIGGERING SAMPLING: sampleN not set ***')

        alias = kwargs.get('alias')
        if alias:
            adf.alias = \
                alias \
                if sampleN \
                else alias + '__AllTaken__NotSampled'

        cache = kwargs.get('cache', True)

        if cache:
            if arimo.debug.ON:
                self.stdout_logger.debug(msg='*** CACHING FOR STREAMING... ***')
                tic = time.time()

            adf.cache(eager=True)

            if arimo.debug.ON:
                toc = time.time()
                self.stdout_logger.debug(
                    msg='*** CACHED FOR STREAMING: {} ROWS   <{:,.1f} m> ***'.format(adf.nRows, (toc - tic) / 60))

        g = adf.toLocalIterator()
        
        while True:
            rows = list(itertools.islice(g, preparedArgs.n))

            while len(rows) < preparedArgs.n:
                g = adf.toLocalIterator()
                rows += list(itertools.islice(g, preparedArgs.n - len(rows)))

            yield [adf._collectCols(
                    rows, cols=cols,
                    asPandas=asPandas,
                    overTime=preparedArgs.colsOverTime[i],
                    padUpToNTimeSteps=preparedArgs.padUpToNTimeSteps[i],
                    padValue=preparedArgs.padValue,
                    padBefore=preparedArgs.padBefore[i])
                   for i, cols in enumerate(preparedArgs.colsLists)] \
                if len(preparedArgs.colsLists) > 1 \
                else adf._collectCols(
                        rows, cols=preparedArgs.colsLists[0],
                        asPandas=asPandas,
                        overTime=preparedArgs.colsOverTime[0],
                        padUpToNTimeSteps=preparedArgs.padUpToNTimeSteps[0],
                        padValue=preparedArgs.padValue,
                        padBefore=preparedArgs.padBefore[0])

    # ****
    # MISC
    # rename
    # split
    # inspectNaNs
    
    def rename(self, **kwargs):
        """
        Return:
            ``SparkADF`` with new column names

        Args:
            **kwargs: arguments of the form ``newColName`` = ``existingColName``
        """
        sparkDF = self._sparkDF
        iCol = self._iCol
        tCol = self._tCol

        for newColName, existingColName in kwargs.items():
            if existingColName not in self._T_AUX_COLS:
                if existingColName == iCol:
                    iCol = newColName

                elif existingColName == tCol:
                    tCol = newColName

                sparkDF = \
                    sparkDF.withColumnRenamed(
                        existing=existingColName,
                        new=newColName)

        return self._decorate(
            obj=sparkDF, nRows=self._cache.nRows,
            iCol=iCol, tCol=tCol)

    def split(self, *weights, **kwargs):
        """
        Split ``SparkADF`` into sub-``SparkADF``'s according to specified weights (which are normalized to sum to 1)

        Return:
            - If ``SparkADF``'s ``.tCol`` property is set, implying the data contains time series, return deterministic
                partitions per ``id`` through time

            - If ``SparkADF`` does not contain time series, randomly shuffle the data and return shuffled sub-``SparkADF``'s

        Args:
            *weights: weights of sub-``SparkADF``'s to split out

            **kwargs:

                - **seed** (int): randomizer seed; *ignored* in the case of deterministic splitting when ``.tCol`` is set
        """
        if (not weights) or weights == (1,):
            return self

        elif self.hasTS:
            nWeights = len(weights)
            cumuWeights = numpy.cumsum(weights) / sum(weights)

            _TS_CHUNK_ID_COL = '__TS_CHUNK_ID__'

            adf = self(
                '*',
                "CONCAT(STRING({}), '---', STRING({})) AS {}"
                    .format(
                        self._PARTITION_ID_COL
                            if self._detPrePartitioned
                            else self._iCol,
                        self._iCol
                            if self._detPrePartitioned
                            else self._T_CHUNK_COL,
                        _TS_CHUNK_ID_COL),
                inheritCache=True,
                inheritNRows=True)

            tsChunkIDs = \
                adf('SELECT \
                        DISTINCT({}) \
                    FROM \
                        this'.format(_TS_CHUNK_ID_COL),
                    inheritCache=False,
                    inheritNRows=False) \
                    .toPandas()[_TS_CHUNK_ID_COL] \
                    .unique().tolist()

            nTSChunkIDs = len(tsChunkIDs)

            if arimo.debug.ON:
                self.stdout_logger.debug(
                    msg='*** SPLITTING BY ID-AND-CHUNK-NUMBER / PARTITION-AND-ID COMBOS ({} COMBOS IN TOTAL) ***'
                        .format(nTSChunkIDs))

            cumuIndices = \
                [0] + \
                [int(round(cumuWeights[i] * nTSChunkIDs))
                 for i in range(nWeights)]

            random.shuffle(tsChunkIDs)

            adfs = [adf.filter(
                        condition=
                            '{} IN ({})'.format(
                                _TS_CHUNK_ID_COL,
                                ', '.join(
                                    "'{0}'".format(_)
                                    for _ in tsChunkIDs[cumuIndices[i]:cumuIndices[i + 1]])))
                        .drop(_TS_CHUNK_ID_COL)
                    for i in range(nWeights)]

        else:
            seed = kwargs.pop('seed', None)

            adfs = [self._decorate(
                        obj=sparkDF,
                        nRows=None,
                        **kwargs)
                    for sparkDF in
                        self._sparkDF.randomSplit(
                            weights=[float(w) for w in weights],
                            seed=seed)]

        for adf in adfs:
            adf._cache.colWidth = copy.deepcopy(self._cache.colWidth)

        return adfs

    def inspectNaNs(self):
        mM = self(
            'SELECT {} FROM this'
                .format(
                    ', '.join(
                        'MIN({0}) AS {0}___min, MAX({0}) AS {0}___max'.format(col)
                        for col in self.columns))) \
            .toPandas().iloc[0]

        df = pandas.DataFrame(
            index=self.columns,
            columns=['min', 'max'])

        colsWithNaNs = []

        for col in self.columns:
            df.at[col, 'min'] = m = mM['{}___min'.format(col)]
            df.at[col, 'max'] = M = mM['{}___max'.format(col)]

            if pandas.notnull(m) and pandas.isnull(M):
                colsWithNaNs.append(col)

        return Namespace(
            df=df,
            colsWithNaNs=colsWithNaNs)
