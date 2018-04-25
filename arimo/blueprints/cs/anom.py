from __future__ import absolute_import

import itertools
import numpy
import os
import shutil
import uuid

from pyspark import SparkFiles
from pyspark.sql.types import DoubleType, StructField, StructType

import arimo.backend
from arimo.blueprints import _docstr_blueprint, _blueprint_from_params
from arimo.blueprints import _PPPBlueprintABC
from arimo.df.spark import ADF
from arimo.util import fs
import arimo.debug

from ..mixins.data_prep import PPPDataPrepMixIn


@_docstr_blueprint
class DLPPPBlueprint(PPPDataPrepMixIn, _PPPBlueprintABC):
    def score(self, *args, **kwargs):
        # scoring batch size
        __batch_size__ = kwargs.pop('__batch_size__', 1000)

        # check if scoring for eval purposes
        __eval__ = kwargs.get(self._MODE_ARG) == self._EVAL_MODE

        adf = self.prep_data(*args, **kwargs)

        assert adf.alias

        if arimo.debug.ON:
            self.stdout_logger.debug(
                '*** SCORING {} WITH BATCH SIZE {} ***'
                    .format(adf, __batch_size__))

        label_var_names = []
        model_file_paths = []
        prep_vec_cols = []
        raw_score_cols = []

        for label_var_name, blueprint_params in self.params.model.component_blueprints.items():
            if (label_var_name in adf.columns) and blueprint_params.model.ver:
                label_var_names.append(label_var_name)

                model_file_path = \
                    os.path.join(
                        _blueprint_from_params(
                            blueprint_params=blueprint_params,
                            aws_access_key_id=self.auth.aws.access_key_id,
                            aws_secret_access_key=self.auth.aws.secret_access_key,
                            verbose=False)
                            .model(ver=blueprint_params.model.ver).dir,
                        blueprint_params.model._persist.file)

                if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                    if model_file_path not in self._MODEL_PATHS_ON_SPARK_WORKER_NODES:
                        _tmp_local_file_name = \
                            str(uuid.uuid4())

                        _tmp_local_file_path = \
                            os.path.join(
                                '/tmp',
                                _tmp_local_file_name)

                        shutil.copyfile(
                            src=model_file_path,
                            dst=_tmp_local_file_path)

                        arimo.backend.spark.sparkContext.addFile(
                            path=_tmp_local_file_path,
                            recursive=False)

                        self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_file_path] = \
                            _tmp_local_file_name   # SparkFiles.get(filename=_tmp_local_file_name)

                    _model_file_path = self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_file_path]

                else:
                    _model_file_path = model_file_path

                model_file_paths.append(_model_file_path)

                prep_vec_cols.append(blueprint_params.data._prep_vec_col + label_var_name)

                raw_score_cols.append(blueprint_params.model.score.raw_score_col_prefix + label_var_name)

        if __eval__:
            adf(self.params.data.id_col,
                *(label_var_names + prep_vec_cols),
                iCol=self.params.data.id_col,
                tCol=None,
                inplace=True)

        def batch(row_iterator_in_partition):
            rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

            while rows:
                yield (rows,) + \
                    tuple(
                        numpy.vstack(
                            row[prep_vec_col]
                            for row in rows)
                        for prep_vec_col in prep_vec_cols)

                rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

        rdd = adf.rdd.mapPartitions(batch)

        def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
            if cluster:
                from dl import _load_keras_model
            else:
                from arimo.util.dl import _load_keras_model

            return [(row[0] +
                         tuple(
                            float(s[0])
                            for s in row[1:]))
                    for row in
                        zip(tup[0],
                            *(_load_keras_model(
                                    file_path=dl_model_file_path)
                                  .predict(
                                    x=x,
                                    batch_size=__batch_size__,
                                    verbose=0)
                              for dl_model_file_path, x in zip(model_file_paths, tup[1:])))]

        return ADF.create(
                data=rdd.flatMap(score),
                schema=StructType(
                    list(adf.schema) +
                    [StructField(
                        name=raw_score_col,
                        dataType=DoubleType(),
                        nullable=True,
                        metadata=None)
                     for raw_score_col in raw_score_cols]),
                samplingRatio=None,
                verifySchema=False,
                iCol=adf.iCol,
                tCol=adf.tCol) \
            .drop(*prep_vec_cols,
                  alias=adf.alias + '__Scored')
