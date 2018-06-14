import keras
import numpy


BATCH_SIZE = 10 ** 3
N_BATCHES_PER_EPOCH = 10 ** 3
N_EPOCHS = 3


GEN_QUEUE_SIZE = 10 ** 2
N_WORKERS = 3


dummy_model = \
    keras.models.Sequential(
        layers=[keras.layers.Dense(
            input_shape=(1,),
            units=1,
            activation='linear',
            use_bias=False)])

dummy_model.compile(
    loss='mse',
    optimizer=keras.optimizers.Nadam())


def dummy_gen():
    while True:
        yield numpy.random.rand(BATCH_SIZE), numpy.random.rand(BATCH_SIZE)


g = dummy_gen()


while True:
    dummy_model.fit_generator(
        generator=g,
        steps_per_epoch=N_BATCHES_PER_EPOCH,
        epochs=N_EPOCHS,
        verbose=1,   # 1 = progress bar, 2 = one line per epoch.
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        class_weight=None,

        max_queue_size=99,
        workers=N_WORKERS,
        use_multiprocessing=True,

        shuffle=False,
        initial_epoch=0)
