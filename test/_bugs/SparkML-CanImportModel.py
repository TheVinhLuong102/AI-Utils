import pandas

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler

from arimo.df.spark import SparkADF


df = pandas.DataFrame(
    data=dict(
        int_x=[1, 2, 3],
        flt_x=[10., 20., 30.]))
adf = SparkADF.create(data=df)
sdf = adf._sparkDF


fltXVectorAssembler = \
    VectorAssembler(
        inputCols=['flt_x'],
        outputCol='_vec_flt_x')
fltXStandardScaler = \
    StandardScaler(
        inputCol='_vec_flt_x',
        outputCol='_stdscl_flt_x',
        withMean=True,
        withStd=True)
fltXStandardScalingPipelineModel = \
    Pipeline(stages=[fltXVectorAssembler, fltXStandardScaler]) \
        .fit(dataset=sdf)
print('\nFloat-X Standard-Scaled Output from {0}:'.format(fltXStandardScalingPipelineModel))
fltXStandardScalingPipelineModel \
    .transform(dataset=sdf) \
    .show()


from _spark_ml_model_to_import import sdf, fltXStandardScalingPipelineModel
print('\nFloat-X Standard-Scaled Output from {0}:'.format(fltXStandardScalingPipelineModel))
fltXStandardScalingPipelineModel \
    .transform(dataset=sdf) \
    .show()
