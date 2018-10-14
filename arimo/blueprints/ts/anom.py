from __future__ import absolute_import

import itertools
import numpy
import os
import shutil
import uuid

from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType

import arimo.backend
from arimo.data.distributed import DDF
from arimo.util import fs
from arimo.util.dl import MASK_VAL
import arimo.debug

from ..base import _blueprint_from_params, AbstractPPPBlueprint, AbstractTimeSerDataPrepMixIn


class DLPPPBlueprint(AbstractTimeSerDataPrepMixIn, AbstractPPPBlueprint):
    _SCORE_ADF_ALIAS = '__Scored__'

    def score(self, *args, **kwargs):
        # scoring batch size
        __batch_size__ = kwargs.pop('__batch_size__', 500)

        # check if scoring for eval purposes
        __eval__ = kwargs.get(self._MODE_ARG) == self._EVAL_MODE

        # whether to cache data at certain stages
        __cache_vector_data__ = kwargs.pop('__cache_vector_data__', False)
        __cache_tensor_data__ = kwargs.pop('__cache_tensor_data__', False)

        # verbosity option
        verbose = kwargs.pop('verbose', True)

        adf = self.prep_data(*args, **kwargs)

        assert isinstance(adf, DDF) and adf.alias

        if arimo.debug.ON:
            self.stdout_logger.debug(
                '*** SCORING {} WITH BATCH SIZE {} ***'
                    .format(adf, __batch_size__))

        label_var_names = []
        model_paths = {}
        prep_vec_cols = {}
        prep_vec_sizes = {}
        args_to_prepare = []
        min_input_ser_len = 0
        max_input_ser_lens = {}
        raw_score_cols = {}

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

                model_paths[label_var_name] = _model_path

                prep_vec_cols[label_var_name] = \
                    _prep_vec_col = \
                    component_blueprint_params.data._prep_vec_col + label_var_name

                prep_vec_sizes[label_var_name] = \
                    component_blueprint_params.data._prep_vec_size

                args_to_prepare.append(
                    (_prep_vec_col, - component_blueprint_params.max_input_ser_len + 1, 0))

                if component_blueprint_params.min_input_ser_len > min_input_ser_len:
                    min_input_ser_len = component_blueprint_params.min_input_ser_len

                max_input_ser_lens[label_var_name] = \
                    component_blueprint_params.max_input_ser_len

                raw_score_cols[label_var_name] = \
                    component_blueprint_params.model.score.raw_score_col_prefix + label_var_name

        if __cache_vector_data__:
            adf.cache(
                eager=True,
                verbose=verbose)

        score_adf = \
            adf._prepareArgsForSampleOrGenOrPred(
                *args_to_prepare,
                n=None,
                fraction=None,
                withReplacement=False,
                seed=None,
                anon=False,
                collect=False,
                pad=MASK_VAL,
                filter='({} >= {})'.format(adf._T_ORD_IN_CHUNK_COL, min_input_ser_len),
                keepOrigRows=False) \
            .adf   # TODO: keepOrigRows should ideally = True, but set to False here

        if arimo.debug.ON:
            self.stdout_logger.debug(
                msg='*** SCORE: PREPARED DDF: {} {} ***\n'
                    .format(adf, adf.columns))

        prep_vec_over_time_cols = {}

        for label_var_name in label_var_names:
            _prep_vec_col = prep_vec_cols[label_var_name]

            prep_vec_over_time_cols[label_var_name] = \
                next(col for col in score_adf.columns
                         if _prep_vec_col in col)

        id_col = str(self.params.data.id_col)
        time_chunk_col = adf._T_CHUNK_COL
        time_ord_in_chunk_col = adf._T_ORD_IN_CHUNK_COL

        def batch(row_iterator_in_partition):
            def _input_tensor(label_var_name, rows):
                prep_vec_col = prep_vec_cols[label_var_name]
                prep_vec_size = prep_vec_sizes[label_var_name]
                prep_vec_over_time_col = prep_vec_over_time_cols[label_var_name]
                max_input_ser_len = max_input_ser_lens[label_var_name]

                return numpy.vstack(
                    numpy.expand_dims(
                        numpy.vstack(
                            [numpy.zeros(
                                (max_input_ser_len - len(row[prep_vec_over_time_col]),
                                prep_vec_size))] +
                            [r[prep_vec_col].toArray()
                             for r in row[prep_vec_over_time_col]]),
                        axis=0)
                    for row in rows)

            rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

            while rows:
                yield ([row[id_col] for row in rows],
                       [row[time_chunk_col] for row in rows],
                       [row[time_ord_in_chunk_col] for row in rows]) + \
                    tuple(_input_tensor(
                            label_var_name=label_var_name,
                            rows=rows)
                          for label_var_name in label_var_names)

                rows = list(itertools.islice(row_iterator_in_partition, __batch_size__))

        rdd = score_adf.rdd.mapPartitions(batch)

        if __cache_tensor_data__:
            rdd.cache()
            rdd.count()

        if component_blueprint_params.model.factory.name.startswith('arimo.dl.experimental.keras'):
            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                if cluster:
                    try:
                        from arimo.util.dl import _load_keras_model

                    except ImportError:
                        from dl import _load_keras_model

                else:
                    from arimo.util.dl import _load_keras_model

                def scores(label_var_name, input_tensor):
                    a = _load_keras_model(
                            file_path=model_paths[label_var_name]) \
                        .predict(
                            x=input_tensor,
                            batch_size=__batch_size__,
                            verbose=0)

                    return [float(s[0])
                            for s in a]

                return zip(tup[0], tup[1], tup[2],
                           *(scores(
                               label_var_name=label_var_name,
                               input_tensor=tup[i + 3])
                               for i, label_var_name in enumerate(label_var_names)))

        else:
            def score(tup, cluster=fs._ON_LINUX_CLUSTER_WITH_HDFS):
                from arimo.util.dl import _load_arimo_dl_model

                def scores(label_var_name, input_tensor):
                    a = _load_arimo_dl_model(
                            dir_path=model_paths[label_var_name],
                            hdfs=cluster) \
                        .predict(
                            data=input_tensor,
                            input_tensor_transform_fn=None,
                            batch_size=__batch_size__)

                    return [float(s[0])
                            for s in a]

                return zip(tup[0], tup[1], tup[2],
                           *(scores(
                               label_var_name=label_var_name,
                               input_tensor=tup[i + 3])
                               for i, label_var_name in enumerate(label_var_names)))

        score_adf = \
            DDF.create(
                data=rdd.flatMap(score),
                schema=StructType(
                    [StructField(
                        name=id_col,
                        dataType=adf._schema[id_col].dataType,
                        nullable=True,
                        metadata=None),
                     StructField(
                        name=time_chunk_col,
                        dataType=IntegerType(),
                        nullable=True,
                        metadata=None),
                     StructField(
                        name=time_ord_in_chunk_col,
                        dataType=IntegerType(),
                        nullable=True,
                        metadata=None)] +
                    [StructField(
                        name=raw_score_cols[label_var_name],
                        dataType=DoubleType(),
                        nullable=True,
                        metadata=None)
                     for label_var_name in label_var_names]),
                samplingRatio=None,
                verifySchema=False)

        score_adf.alias = self._SCORE_ADF_ALIAS

        return adf(
            'SELECT \
                {4}, \
                {5} \
            FROM \
                this LEFT JOIN {0} \
                    ON this.{1} = {0}.{1} AND \
                       this.{2} = {0}.{2} AND \
                       this.{3} = {0}.{3}'
            .format(
                self._SCORE_ADF_ALIAS,
                id_col,
                time_chunk_col,
                time_ord_in_chunk_col,
                ', '.join(
                    'this.{0} AS {0}'.format(col)
                        for col in
                            (([id_col] + label_var_names)
                             if __eval__
                             else set(adf.columns).difference(prep_vec_cols.values()))),
                ', '.join(
                    '{0}.{1} AS {1}'.format(self._SCORE_ADF_ALIAS, col)
                    for col in score_adf.columns[3:])),
            alias=adf.alias + '__Scored')
