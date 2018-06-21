import os
import numpy

import arimo.backend
from arimo.util import fs


TMP_DIR_PATH = '/tmp'
DUMMY_MODEL_FILE_NAME = 'dummy-Keras-model.h5'


dummy_model_file_path = \
    os.path.join(
        TMP_DIR_PATH,
        DUMMY_MODEL_FILE_NAME)


arimo.backend.keras.models.Sequential(
    layers=[arimo.backend.keras.layers.Dense(
                input_shape=(1,),
                units=1,
                activation='linear',
                use_bias=False)]) \
    .save(
        filepath=dummy_model_file_path,
        overwrite=True,
        include_optimizer=False)


arimo.backend.initSpark(sparkApp='test')


arimo.backend.spark.sparkContext.addFile(
    path=dummy_model_file_path,
    recursive=False)


def score(x, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS, dummy_model_file_name=DUMMY_MODEL_FILE_NAME):
    if cluster:
        from dl import _load_keras_model
    else:
        from arimo.util.dl import _load_keras_model

    return _load_keras_model(file_path=dummy_model_file_name) \
        .predict(x=x, verbose=0)


print(arimo.backend.spark.sparkContext
        .parallelize(numpy.array([[i]]) for i in range(27))
        .map(score)
        .collect())
