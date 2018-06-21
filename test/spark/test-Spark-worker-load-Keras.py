from __future__ import absolute_import

import os
import numpy

import arimo.backend
from arimo.util import fs
from arimo.util.dl import _LOCAL_MODELS_DIR


_TMP_DIR_PATH = '/tmp'
_DUMMY_MODEL_FILE_NAME = 'dummy-Keras-model.h5'
_DUMMY_MODEL_FILE_PATH = \
    os.path.join(
        _TMP_DIR_PATH,
        _DUMMY_MODEL_FILE_NAME)


dummy_model = \
    arimo.backend.keras.models.Sequential(
        layers=[arimo.backend.keras.layers.Dense(
                    input_shape=(1,),
                    units=1,
                    activation='linear',
                    use_bias=False)])

dummy_model.compile(
    loss='mse',
    optimizer=arimo.backend.keras.optimizers.Nadam())


dummy_model.save(
    filepath=_DUMMY_MODEL_FILE_PATH,
    overwrite=True,
    include_optimizer=True)


if fs._ON_LINUX_CLUSTER_WITH_HDFS:
    fs.put(
        from_local=_DUMMY_MODEL_FILE_PATH,
        to_hdfs=_DUMMY_MODEL_FILE_PATH,
        is_dir=False,
        _mv=True)


arimo.backend.initSpark(
    sparkApp='test')


def rm_existing_model(_):
    import fs

    fs.rm(path=os.path.join(
            _LOCAL_MODELS_DIR,
            _DUMMY_MODEL_FILE_PATH.strip(os.sep)),
          is_dir=False,
          hdfs=False)


def score(x, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
    if cluster:
        from dl import _load_keras_model

    else:
        from arimo.util.dl import _load_keras_model

    return _load_keras_model(
            file_path=_DUMMY_MODEL_FILE_PATH,
            hdfs=cluster) \
        .predict(
            x=x,
            verbose=0)


rdd = arimo.backend.spark.sparkContext.parallelize(numpy.array([[i]]) for i in range(27))

if fs._ON_LINUX_CLUSTER_WITH_HDFS:
    rdd.map(rm_existing_model).collect()

print(rdd.map(score).collect())
