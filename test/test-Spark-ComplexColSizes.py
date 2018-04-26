import pandas

from pyspark.ml.feature import VectorAssembler, VectorSizeHint
from pyspark.ml import PipelineModel

from arimo.df.spark import ADF


adf = ADF.create(
    data=pandas.DataFrame(
        data=dict(
            x=[0],
            y=[1],
            z=[2])))(
    '*',
    'ARRAY(x) AS Ax',
    'ARRAY(y, z) AS Ayz',
    'STRUCT(x) AS Sx',
    'STRUCT(y, z) AS Syz')(
        PipelineModel(
            stages=[
                VectorAssembler(inputCols=['x'], outputCol='Vx'),
                VectorAssembler(inputCols=['y', 'z'], outputCol='Vyz')]).transform)


vadf = adf(
    PipelineModel(
        stages=[
            # VectorAssembler(inputCols=['Ax', 'Ayz'], outputCol='VAxAyz'),   # ArrayType(...) not supported
            # VectorAssembler(inputCols=['Sx', 'Syz'], outputCol='VSxSyz'),   # StructType(...) of not supported
            VectorAssembler(inputCols=['Vx', 'Vyz'], outputCol='VVxVyz')
        ]).transform)
