from jpmml_sparkml import toPMMLBytes

import arimo.backend


def jpmml_bytes(df, sparkMLModel):
    return toPMMLBytes(
        sc=arimo.backend.spark.sparkContext,
        df=df,
        pipelineModel=sparkMLModel)
