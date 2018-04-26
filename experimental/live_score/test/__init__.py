import pandas

from pyspark.ml import Pipeline
from pyspark.ml.feature import \
    Binarizer, \
    Bucketizer, \
    QuantileDiscretizer, \
    StringIndexer, \
    IndexToString, \
    OneHotEncoder, \
    VectorIndexer, \
    \
    MaxAbsScaler, \
    MinMaxScaler, \
    Normalizer, \
    StandardScaler, \
    \
    VectorAssembler

from arimo.df.spark import ADF
from arimo.util import Namespace


def data(spark_data=True):
    d = dict(
        int_x=1,
        flt_x=10.)

    namespace = Namespace(
        int_x=2,
        flt_x=20.)

    df = pandas.DataFrame(
        data=dict(
            int_x=[1, 2, 3],
            flt_x=[10., 20., 30.]))

    if spark_data:
        adf = ADF.create(data=df)
        sdf = adf._sparkDF

        fltXVectorAssembler = \
            VectorAssembler(
                inputCols=['flt_x'],
                outputCol='_vec_flt_x')
        fltXVectorAssemblingPipelineModel = \
            Pipeline(stages=[fltXVectorAssembler]) \
            .fit(dataset=sdf)
        fltXStandardScaler = \
            StandardScaler(
                inputCol='_vec_flt_x',
                outputCol='_stdscl_flt_x',
                withMean=True,
                withStd=True)
        fltXStandardScalingPipelineModel = \
            Pipeline(stages=[fltXVectorAssembler, fltXStandardScaler]) \
            .fit(dataset=sdf)

        intXOneHotEncoder = \
            OneHotEncoder(
                inputCol='int_x',
                outputCol='_ohe_int_x',
                # includeFirst=True,   # unclear in Spark doc
                dropLast=True)
        intXOneHotEncodingPipelineModel = \
            Pipeline(stages=[intXOneHotEncoder]) \
            .fit(dataset=sdf)

        return Namespace(
            dict=d, namespace=namespace, df=df, adf=adf, sdf=sdf,
            fltXVectorAssembler=fltXVectorAssembler,
            fltXVectorAssemblingPipelineModel=fltXVectorAssemblingPipelineModel,
            fltXStandardScalingPipelineModel=fltXStandardScalingPipelineModel,
            intXOneHotEncodingPipelineModel=intXOneHotEncodingPipelineModel)

    else:
        return Namespace(
            dict=d, namespace=namespace, df=df)
