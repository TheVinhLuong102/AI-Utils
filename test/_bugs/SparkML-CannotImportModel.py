from _spark_ml_model_to_import import sdf, fltXStandardScalingPipelineModel


fltXStandardScalingPipelineModel \
    .transform(dataset=sdf) \
    .show()
