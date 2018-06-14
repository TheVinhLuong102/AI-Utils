from __future__ import absolute_import

import numpy

import arimo.backend


x = y = numpy.array([[-3], [-2], [-1], [1], [2], [3]])



# *** BAD!!! ***
model___merge___reloc = \
  arimo.backend.keras.models.Sequential(
    layers=[arimo.backend.keras.layers.Dense(
      input_shape=(1,),
      units=1,
      activation='linear',
      use_bias=True)])

y0 = model___merge___reloc.predict(x)

multi_gpu_model___merge___reloc = \
    arimo.backend.keras.utils.multi_gpu_model(
        model___merge___reloc,
        gpus=2,
        cpu_merge=True,
        cpu_relocation=True)

multi_gpu_model___merge___reloc.compile(
    loss='mae',
    optimizer=arimo.backend.keras.optimizers.Nadam())

multi_gpu_model___merge___reloc.fit(
    x=x, y=y,
    batch_size=1,
    epochs=100,
    verbose=2)

y1 = model___merge___reloc.predict(x)

print(numpy.hstack((y0 - y, y1 - y)))


# *** BAD ***
model___merge___no_reloc = \
    arimo.backend.keras.models.Sequential(
        layers=[arimo.backend.keras.layers.Dense(
            input_shape=(1,),
            units=1,
            activation='linear',
            use_bias=True)])

y0 = model___merge___no_reloc.predict(x)

multi_gpu_model___merge___no_reloc = \
    arimo.backend.keras.utils.multi_gpu_model(
        model___merge___no_reloc,
        gpus=2,
        cpu_merge=True,
        cpu_relocation=False)

multi_gpu_model___merge___no_reloc.compile(
    loss='mae',
    optimizer=arimo.backend.keras.optimizers.Nadam())

multi_gpu_model___merge___no_reloc.fit(
    x=x, y=y,
    batch_size=1,
    epochs=9,
    verbose=1)

y1 = model___merge___no_reloc.predict(x)

print(numpy.hstack((y0 - y, y1 - y)))


# *** BAD!!! ***
model___no_merge___reloc = \
    arimo.backend.keras.models.Sequential(
        layers=[arimo.backend.keras.layers.Dense(
            input_shape=(1,),
            units=1,
            activation='linear',
            use_bias=True)])

multi_gpu_model___no_merge___reloc = \
    arimo.backend.keras.utils.multi_gpu_model(
        model___no_merge___reloc,
        gpus=2,
        cpu_merge=True,
        cpu_relocation=False)

multi_gpu_model___no_merge___reloc.compile(
    loss='mae',
    optimizer=arimo.backend.keras.optimizers.Nadam())

multi_gpu_model___no_merge___reloc.fit(
    x=x, y=y,
    batch_size=1,
    epochs=9,
    verbose=1)

print(model___no_merge___reloc.predict(x))


# *** BAD!!! ***
model___no_merge___no_reloc = \
    arimo.backend.keras.models.Sequential(
        layers=[arimo.backend.keras.layers.Dense(
            input_shape=(1,),
            units=1,
            activation='linear',
            use_bias=True)])

multi_gpu_model___no_merge___no_reloc = \
    arimo.backend.keras.utils.multi_gpu_model(
        model___no_merge___no_reloc,
        gpus=2,
        cpu_merge=True,
        cpu_relocation=False)

multi_gpu_model___no_merge___no_reloc.compile(
    loss='mae',
    optimizer=arimo.backend.keras.optimizers.Nadam())

multi_gpu_model___no_merge___no_reloc.fit(
    x=x, y=y,
    batch_size=1,
    epochs=9,
    verbose=1)

print(model___no_merge___no_reloc.predict(x))
