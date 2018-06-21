from __future__ import absolute_import

import numpy

import arimo.backend


x = y = numpy.array([[-3], [-2], [-1], [1], [2], [3]])


def train(merge=False, reloc=False):
    model = arimo.backend.keras.models.Sequential(
        layers=[arimo.backend.keras.layers.Dense(
                    input_shape=(1,),
                    units=1,
                    activation='linear',
                    use_bias=True,
                    kernel_initializer='zeros',
                    bias_initializer='ones')])

    y0 = model.predict(x)

    multi_gpu_model = \
        arimo.backend.keras.utils.multi_gpu_model(
            model,
            gpus=2,
            cpu_merge=merge,
            cpu_relocation=reloc)

    multi_gpu_model.compile(
        loss='mae',
        optimizer=arimo.backend.keras.optimizers.Nadam())

    hist = multi_gpu_model.fit(
        x=x, y=y,
        batch_size=1,
        epochs=100,
        verbose=0)

    y1 = model.predict(x)
    y2 = multi_gpu_model.predict(x)

    print(hist.history['loss'][-1])
    return numpy.hstack((y0, y1, y2)) - y


print(train(merge=False, reloc=False))
print(train(merge=True, reloc=False))
print(train(merge=False, reloc=True))
print(train(merge=True, reloc=True))
