import numpy

import arimo.backend


MODEL_O_FILE_PATH = '/tmp/Keras-Model-0.h5'
MODEL_1_FILE_PATH = '/tmp/Keras-Model-1.h5'
MODEL_2_FILE_PATH = '/tmp/Keras-Model-2.h5'


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

    model.save(
        filepath=MODEL_O_FILE_PATH,
        overwrite=True,
        include_optimizer=True)
    model_0 = \
        arimo.backend.keras.models.load_model(
            filepath=MODEL_O_FILE_PATH,
            custom_objects=None,
            compile=True)
    y0 = model_0.predict(x)

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

    model.save(
        filepath=MODEL_1_FILE_PATH,
        overwrite=True,
        include_optimizer=True)
    model_1 = \
        arimo.backend.keras.models.load_model(
            filepath=MODEL_1_FILE_PATH,
            custom_objects=None,
            compile=True)
    y1 = model_1.predict(x)

    multi_gpu_model.save(
        filepath=MODEL_2_FILE_PATH,
        overwrite=True,
        include_optimizer=True)
    model_2 = \
        arimo.backend.keras.models.load_model(
            filepath=MODEL_2_FILE_PATH,
            custom_objects=None,
            compile=True)
    y2 = model_2.predict(x)

    print('MERGE= {}, RELOC= {}: Loss={}'.format(
        merge, reloc, hist.history['loss'][-1]))
    return numpy.hstack((y0, y1, y2)) - y


print(train(merge=False, reloc=False))
print(train(merge=True, reloc=False))
print(train(merge=False, reloc=True))
print(train(merge=True, reloc=True))
