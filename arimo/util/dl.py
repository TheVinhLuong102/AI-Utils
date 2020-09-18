import os
import keras

try:
    from . import fs
except:
    # use relative import to avoid "ValueError: Attempted relative import in non-package" in Spark workers
    import fs


MASK_VAL = .13 ** 3


_LOCAL_MODELS_DIR = '/tmp/.arimo/models'

if not os.path.isdir(_LOCAL_MODELS_DIR):
    fs.mkdir(
        dir=_LOCAL_MODELS_DIR,
        hdfs=False)


_LOADED_MODELS = {}


def _load_keras_model(file_path):
    global _LOADED_MODELS

    if file_path not in _LOADED_MODELS:
        assert os.path.isfile(file_path), \
            f'*** {file_path} FILE DOES NOT EXIST ***'

        _LOADED_MODELS[file_path] = \
            keras.models.load_model(
                file_path,
                custom_objects=None,
                compile=True,
                options=None)

    return _LOADED_MODELS[file_path]
