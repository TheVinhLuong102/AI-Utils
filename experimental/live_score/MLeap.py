from __future__ import print_function

import mleap.pyspark.spark_support

from __init__ import data


_MLEAP_BUNDLE_FILE_PATH = '/tmp/testMLeapBundle.zip'


data = data()

print(data.sdf)
data.sdf.show()


fltXVectorAssembledDF = \
    data.fltXVectorAssembler \
        .transform(dataset=data.sdf)

fltXVectorAssemblingPipelinedDF = \
    data.fltXVectorAssemblingPipelineModel \
        .transform(dataset=data.sdf)

print('\nFloat-X Vector-Assembled Output:')
fltXVectorAssembledDF.show()
fltXVectorAssemblingPipelinedDF.show()

print('Converting Float-X Vector-Assembler to MLeap Bundle... ', end='')
try:
    data.fltXVectorAssembler.serializeToBundle(
        path=_MLEAP_BUNDLE_FILE_PATH,
        dataset=fltXVectorAssembledDF)
    print('done!')
except Exception as err:
    print('FAILED!')
    print(err)

print('Converting Float-X Vector-Assembling Model to MLeap Bundle... ', end='')
try:
    data.fltXVectorAssemblingPipelineModel.serializeToBundle(
        path=_MLEAP_BUNDLE_FILE_PATH,
        dataset=fltXVectorAssemblingPipelinedDF)
    print('done!')
except Exception as err:
    print('FAILED!')
    print(err)


fltXStandardScalingPipelinedDF = \
    data.fltXStandardScalingPipelineModel \
        .transform(dataset=data.sdf)

print('\nFloat-X Standard-Scaled Output:')
fltXStandardScalingPipelinedDF.show()

print('Converting Float-X Standard-Scaling Model to MLeap Bundle... ', end='')
try:
    data.fltXStandardScalingPipelineModel.serializeToBundle(
        path=_MLEAP_BUNDLE_FILE_PATH,
        dataset=fltXStandardScalingPipelinedDF)
    print('done!')
except Exception as err:
    print('FAILED!')
    print(err)


intXOneHotEncodingPipelinedDF = \
    data.intXOneHotEncodingPipelineModel \
        .transform(dataset=data.sdf)

print('\nInt-X One-Hot-Encoded Output:')
intXOneHotEncodingPipelinedDF.show()

print('Converting Int-X One-Hot-Encoding Model to MLeap Bundle... ', end='')
try:
    data.intXOneHotEncodingPipelineModel.serializeToBundle(
        path=_MLEAP_BUNDLE_FILE_PATH,
        dataset=intXOneHotEncodingPipelinedDF)
    print('done!')
except Exception as err:
    print('FAILED!')
    print(err)
