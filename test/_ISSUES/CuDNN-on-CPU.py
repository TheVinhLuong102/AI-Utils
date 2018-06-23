from __future__ import print_function

import numpy

import arimo.backend


MODEL_FILE_PATH = '/tmp/CuDNN.h5'


x = numpy.array([[[-3]], [[-2]], [[-1]], [[1]], [[2]], [[3]]])

y = numpy.array([[-3], [-2], [-1], [1], [2], [3]])


model = arimo.backend.keras.models.Sequential(
            layers=[arimo.backend.keras.layers.CuDNNLSTM(
                        input_shape=(1, 1),
                        units=1)])

model.compile(
    loss='mae',
    optimizer=arimo.backend.keras.optimizers.Nadam())

hist = model.fit(
    x=x, y=y,
    batch_size=1,
    epochs=100,
    verbose=1)

print(model.predict(x=x))

model.save(
    filepath=MODEL_FILE_PATH,
    overwrite=True,
    include_optimizer=True)


# CANNOT SIMPLY LOAD CuDNN-TRAINED MODEL USING CPU
# https://stackoverflow.com/questions/48086014/keras-model-with-cudnnlstm-layers-doesnt-work-on-production-server
# https://stackoverflow.com/questions/50127488/tensorflow-use-model-trained-in-cudnnlstm-in-cpu
loaded_model = \
    arimo.backend.keras.models.load_model(
        filepath=MODEL_FILE_PATH,
        custom_objects=None,
        compile=True)
