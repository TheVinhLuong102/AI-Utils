from __future__ import absolute_import

import itertools
import numpy
import os
import shutil
import uuid

from pyspark.sql.types import DoubleType, StructField, StructType

import arimo.backend
from arimo.data.distributed import DDF
from arimo.util import fs
import arimo.debug

from .. import _blueprint_from_params, AbstractPPPBlueprint


class DLPPPBlueprint(AbstractPPPBlueprint):
    def score(self, *args, **kwargs):
        # scoring batch size
        __batch_size__ = kwargs.pop('__batch_size__', 1000)

        # check if scoring for eval purposes
        __eval__ = kwargs.get(self._MODE_ARG) == self._EVAL_MODE

        adf = self.prep_data(*args, **kwargs)

        assert (isinstance(adf, DDF) or hasattr(adf, '_sparkDF')) and adf.alias

        if arimo.debug.ON:
            self.stdout_logger.debug(
                '*** SCORING {} WITH BATCH SIZE {} ***'
                    .format(adf, __batch_size__))

        label_var_names = []
        model_paths = []
        prep_vec_cols = []
        raw_score_cols = []

        for label_var_name, component_blueprint_params in self.params.model.component_blueprints.items():
            if (label_var_name in adf.columns) and component_blueprint_params.model.ver:
                label_var_names.append(label_var_name)

                # legacy fix
                _component_blueprint_uuid_prefix = \
                    '{}---{}'.format(self.params.uuid, label_var_name)

                if not component_blueprint_params.uuid.startswith(_component_blueprint_uuid_prefix):
                    component_blueprint_params.uuid = _component_blueprint_uuid_prefix

                component_blueprint = \
                    _blueprint_from_params(
                        blueprint_params=component_blueprint_params,
                        aws_access_key_id=self.auth.aws.access_key_id,
                        aws_secret_access_key=self.auth.aws.secret_access_key,
                        verbose=False)

                assert component_blueprint.params.uuid == component_blueprint_params.uuid, \
                    '*** {} ***'.format(component_blueprint.params.uuid)

                if component_blueprint_params.model.factory.name.startswith('arimo.dl.experimental.keras'):
                    model_path = \
                        os.path.join(
                            component_blueprint.model(ver=component_blueprint_params.model.ver).dir,
                            component_blueprint_params.model._persist.file)

                    assert os.path.isfile(model_path), \
                        '*** {} DOES NOT EXIST ***'.format(model_path)

                    if fs._ON_LINUX_CLUSTER_WITH_HDFS:
                        if model_path not in self._MODEL_PATHS_ON_SPARK_WORKER_NODES:
                            _tmp_local_file_name = \
                                str(uuid.uuid4())

                            _tmp_local_file_path = \
                                os.path.join(
                                    '/tmp',
                                    _tmp_local_file_name)

                            shutil.copyfile(
                                src=model_path,
                                dst=_tmp_local_file_path)

                            arimo.backend.spark.sparkContext.addFile(
                                path=_tmp_local_file_path,
                                recursive=False)

                            self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_path] = \
                                _tmp_local_file_name   # SparkFiles.get(filename=_tmp_local_file_name)

                        _model_path = self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_path]

                    else:
                        _model_path = model_path

                else:
                    model_path = _model_path = \
                        component_blueprint.model(ver=component_blueprint_params.model.ver).dir

                    assert os.path.isdir(model_path), \
                        '*** {} DOES NOT EXIST ***'.format(model_path)

                    if fs._ON_LINUX_CLUSTER_WITH_HDFS and (model_path not in self._MODEL_PATHS_ON_SPARK_WORKER_NODES):
                        fs.put(
                            from_local=model_path,
                            to_hdfs=model_path,
                            is_dir=True,
                            _mv=False)

                        self._MODEL_PATHS_ON_SPARK_WORKER_NODES[model_path] = model_path

                model_paths.append(_model_path)

                prep_vec_cols.append(component_blueprint_params.data._prep_vec_col + label_var_name)

                raw_score_cols.append(component_blueprint_params.model.score.raw_score_col_prefix + label_var_name)

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

        if component_blueprint_params.model.factory.name.startswith('arimo.dl.experimental.keras'):
            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                if cluster:
                    try:
                        from arimo.util.dl import _load_keras_model

                    except ImportError:
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
                                        file_path=model_path)
                                    .predict(
                                        x=x,
                                        batch_size=__batch_size__,
                                        verbose=0)
                                  for model_path, x in zip(model_paths, tup[1:])))]

        else:
            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                from arimo.util.dl import _load_arimo_dl_model

                return [(row[0] +
                            tuple(
                                float(s[0])
                                for s in row[1:]))
                        for row in
                            zip(tup[0],
                                *(_load_arimo_dl_model(
                                        dir_path=model_path,
                                        hdfs=cluster)
                                    .predict(
                                        data=x,
                                        input_tensor_transform_fn=None,
                                        batch_size=__batch_size__)
                                  for model_path, x in zip(model_paths, tup[1:])))]

        return DDF.create(
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
