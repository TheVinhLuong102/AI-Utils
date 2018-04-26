from __future__ import absolute_import, division, print_function

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy
import six
import tensorflow as tf
from tensorflow.python.ops import random_ops

from arimo.backend import _TF_CONFIG
from arimo.dl.experimental.keras import _TFGraphOp, _TFGraphLayer, _TFGraphModel


# CONSTANTS
CROSSSECT_TENSOR_NAME = 'crosssect'
TIMESER_TENSOR_NAME = 'timeser'
OUTPUT_TENSOR_NAME = 'y'

GRAPH_0___CROSSSECT_INPUT_SIZE = 3
GRAPH_0___TIMESER_INPUT_SIZES = (
  6,   # timesteps
  9    # no. of vars
)

GRAPH_1___CROSSSECT_INPUT_SIZE = 9
GRAPH_1___TIMESER_INPUT_SIZES = (
  6,   # timesteps
  3    # no. of vars
)


# create empty TF graphs
tf_graph_0 = tf.Graph()
tf_graph_1 = tf.Graph()


# create full TF graphs
with tf_graph_0.as_default():
  crosssectX_0 = \
    tf.placeholder(
      dtype=tf.float32,
      shape=(GRAPH_0___CROSSSECT_INPUT_SIZE,),
      name=CROSSSECT_TENSOR_NAME)
  timeserX_0 = \
    tf.placeholder(
      dtype=tf.float32,
      shape=GRAPH_0___TIMESER_INPUT_SIZES,
      name=TIMESER_TENSOR_NAME)
  y_0 = \
    tf.concat(
      values=[
        crosssectX_0,
        tf.reduce_mean(
          input_tensor=timeserX_0,
          axis=0,
          keep_dims=False,
          name=None,
          reduction_indices=None)
      ],
      axis=0,
      name=None
    ) * \
    tf.Variable(
      initial_value=
        random_ops.random_uniform(
          shape=(),
          minval=0,
          maxval=1,
          dtype=tf.float32,
          seed=None,
          name=None
        ),
      trainable=True,
      collections=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=tf.float32,
      expected_shape=()
    ) + \
    tf.Variable(
      initial_value=
        random_ops.random_uniform(
          shape=(),
          minval=0,
          maxval=1,
          dtype=tf.float32,
          seed=None,
          name=None
        ),
      trainable=True,
      collections=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=tf.float32,
      expected_shape=()
    )

with tf_graph_1.as_default():
  crosssectX_1 = \
    tf.placeholder(
      dtype=tf.float32,
      shape=(GRAPH_1___CROSSSECT_INPUT_SIZE,),
      name=CROSSSECT_TENSOR_NAME)
  timeserX_1 = \
    tf.placeholder(
      dtype=tf.float32,
      shape=GRAPH_1___TIMESER_INPUT_SIZES,
      name=TIMESER_TENSOR_NAME)
  y_1 = \
    tf.concat(
      values=[
        crosssectX_1,
        tf.reduce_mean(
          input_tensor=timeserX_1,
          axis=0,
          keep_dims=False,
          name=None,
          reduction_indices=None)],
      axis=0,
      name=None
    ) * \
    tf.Variable(
      initial_value=
        random_ops.random_uniform(
          shape=(),
          minval=0,
          maxval=1,
          dtype=tf.float32,
          seed=None,
          name=None
        ),
      trainable=True,
      collections=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=tf.float32,
      expected_shape=()
    ) + \
    tf.Variable(
      initial_value=
        random_ops.random_uniform(
          shape=(),
          minval=0,
          maxval=1,
          dtype=tf.float32,
          seed=None,
          name=None
        ),
      trainable=True,
      collections=None,
      validate_shape=True,
      caching_device=None,
      name=None,
      variable_def=None,
      dtype=tf.float32,
      expected_shape=()
    )


# create TF sessions
tf_session_0 = \
  tf.Session(
    target='',
      # The execution engine to connect to. Defaults to using an in-process engine.

    graph=tf_graph_0,
      # The Graph to be launched

    config=_TF_CONFIG
      # protocol buffer with configuration options for the session.
  )

tf_session_1 = \
  tf.Session(
    target='',
      # The execution engine to connect to. Defaults to using an in-process engine.

    graph=tf_graph_1,
      # The Graph to be launched

    config=_TF_CONFIG
      # protocol buffer with configuration options for the session.
  )


# create Keras models
with tf_session_0.as_default():
  tf_graph_op_0 = \
    _TFGraphOp(
      session=tf_session_0,
      inputs=(crosssectX_0, timeserX_0),
      outputs=y_0)
  tf_graph_layer_0 = \
    _TFGraphLayer(
      session=tf_session_0,
      inputs=(crosssectX_0, timeserX_0),
      outputs=y_0)

with tf_session_1.as_default():
  tf_graph_op_1 = \
    _TFGraphOp(
      session=tf_session_1,
      inputs=(crosssectX_1.name, timeserX_1.name),
      outputs=y_1.name)
  tf_graph_layer_1 = \
    _TFGraphLayer(
      session=tf_session_1,
      inputs=(crosssectX_1.name, timeserX_1.name),
      outputs=y_1.name)
