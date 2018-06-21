from __future__ import absolute_import

from pprint import pprint

import arimo.backend


def import_deps(_):
    d = {}

    try:
        import tensorflow
        d['TF'] = tensorflow.__version__
    except ImportError:
        d['TF'] = None

    try:
        import keras
        d['Keras'] = keras.__version__
    except ImportError:
        d['Keras'] = None

    try:
        import h5py
        d['H5'] = h5py.__version__
    except ImportError:
        d['H5'] = None
    
    return d


arimo.backend.initSpark(sparkApp='test')

pprint(arimo.backend.spark.sparkContext
        .parallelize(range(27))
        .map(import_deps)
        .collect())
