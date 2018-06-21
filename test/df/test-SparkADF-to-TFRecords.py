from __future__ import print_function

import os
import pandas

from pyspark.ml.feature import VectorAssembler

from google.protobuf.json_format import MessageToJson
import tensorflow as tf

from arimo.df.spark import SparkADF
from arimo.util import fs


TFR_PATH = '/tmp/.adf.tfr'


adf = SparkADF.create(
    data=pandas.DataFrame(
        data=dict(
            x=[-1, 0, 1],
            y=[-3., 0., 3.])))

adf('*',
    'ARRAY(x, y) AS a',
    inplace=True)

adf(VectorAssembler(
        inputCols=['x', 'y'],
        outputCol='v').transform,
    inplace=True)

adf('SELECT \
        *, \
        COLLECT_LIST(x) OVER w AS x_seq, \
        COLLECT_LIST(y) OVER w AS y_seq, \
        COLLECT_LIST(a) OVER w AS a_seq, \
        COLLECT_LIST(v) OVER w AS v_seq \
    FROM \
        this \
    WINDOW \
        w AS (ORDER BY x)',
    inplace=True)

adf.show()


fs.rm(
    path=TFR_PATH,
    hdfs=fs._ON_LINUX_CLUSTER_WITH_HDFS,
    is_dir=True)

adf['x', 'y', 'a',
    # 'v',
    'x_seq', 'y_seq',
    # 'a_seq',
    # 'v_seq'
].save(
    path=TFR_PATH,
    format='tfrecords',
    verbose=True)


if fs._ON_LINUX_CLUSTER_WITH_HDFS:
    pass

else:
    for file_name in os.listdir(TFR_PATH):
        if file_name[0].isalpha():
            file_path = os.path.join(TFR_PATH, file_name)

            if os.path.getsize(filename=file_path):
                for example in tf.python_io.tf_record_iterator(path=file_path):
                    print(MessageToJson(tf.train.Example.FromString(example)))
