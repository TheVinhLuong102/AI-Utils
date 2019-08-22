from __future__ import absolute_import, division

import copy
import math
import pandas
import random

import arimo.backend
from arimo.data.parquet import S3ParquetDataFeeder
from arimo.data.distributed_parquet import S3ParquetDistributedDataFrame
from arimo.dl.base import LossPlateauLrDecay
from arimo.util import Namespace
import arimo.debug

from .. import BlueprintedArimoDLModel, BlueprintedKerasModel, RegrEvalMixIn
from . import AbstractDLCrossSectSupervisedBlueprint


class DLBlueprint(RegrEvalMixIn, AbstractDLCrossSectSupervisedBlueprint):
    _DEFAULT_PARAMS = \
        copy.deepcopy(
            AbstractDLCrossSectSupervisedBlueprint._DEFAULT_PARAMS)

    _DEFAULT_PARAMS.update(
        model=Namespace(
            factory=Namespace(
                name='arimo.dl.cross_sectional.FfnResnetRegressor'),

            train=Namespace(
                objective=None   # *** DON'T IMPOSE MSE UP-FRONT AS IT'S OVER-SENSITIVE TO LARGE OUTLIERS ***
            )),

        __metadata__={
            'model': Namespace(
                label='Model Params'),

            'model.factory': Namespace(
                label='Model-Initializing Factory Function Name & Params')})

    def train(self, *args, **kwargs):
        __gen_queue_size__ = \
            kwargs.pop(
                '__gen_queue_size__',
                self.DEFAULT_MODEL_TRAIN_MAX_GEN_QUEUE_SIZE)
        assert __gen_queue_size__, \
            '*** __gen_queue_size__ = {} ***'.format(__gen_queue_size__)

        __n_workers__ = \
            kwargs.pop(
                '__n_workers__',
                self.DEFAULT_MODEL_TRAIN_N_WORKERS)
        assert __n_workers__, \
            '*** __n_workers__ = {} ***'.format(__n_workers__)

        __n_gpus__ = \
            kwargs.pop(
                '__n_gpus__',
                self.DEFAULT_MODEL_TRAIN_N_GPUS)
        assert __n_gpus__, \
            '*** __n_gpus__ = {} ***'.format(__n_gpus__)

        __cpu_merge__ = bool(kwargs.pop('__cpu_merge__', True))
        __cpu_reloc__ = bool(kwargs.pop('__cpu_reloc__', False))   # *** cpu_relocation MAKES TEMPLATE MODEL WEIGHTS FAIL TO UPDATE ***

        # verbosity
        verbose = kwargs.pop('verbose', True)

        # *** MUST .prep_data(...) FIRST, AS IT ALTERS CERTAIN BLUEPRINT PARAMS ***
        adf = self.prep_data(
            __mode__=self._TRAIN_MODE,
            verbose=verbose,
            *args, **kwargs)

        self._derive_model_train_params(
            data_size=
                (adf.approxNRows
                 if isinstance(adf, S3ParquetDataFeeder)
                 else adf.nRows)
                if self.params.model.train.n_samples_max_multiple_of_data_size
                else None)

        filter = \
            {self.params.data.label.var:
                 (self.params.data.label.lower_outlier_threshold,
                  self.params.data.label.upper_outlier_threshold)} \
            if self.params.data.label.excl_outliers and \
               self.params.data.label.outlier_tails and \
               self.params.data.label.outlier_tail_proportion and \
               (self.params.data.label.outlier_tail_proportion < .5) and \
               (pandas.notnull(self.params.data.label.lower_outlier_threshold) or
                pandas.notnull(self.params.data.label.upper_outlier_threshold)) \
            else {}

        model = self.model(ver=None) \
            if self.params.model.ver is None \
            else self.model(ver=self.params.model.ver).copy()

        model.stdout_logger.info(
            'TRAINING:'
            '\n- Pred Vars Vec Size: {:,}'
            '\n- Train Samples: {:,}'
            '\n- Val Samples: {:,}'
            '\n- Epochs: {:,}'
            '\n- Train Samples/Epoch: {:,}'
            '\n- Train Batch Size: {:,}'
            '\n- Val Samples/Epoch: {:,}'
            '\n- Val Batch Size: {:,}'
            '\n- Generator Queue Size: {:,}'
            '\n- Processes/Threads: {:,}'
            '\n- GPUs: {}{}'
            '\n- Filter: {}'
            .format(
                self.params.data._prep_vec_size,
                self.params.model.train._n_train_samples,
                self.params.model.train._n_val_samples,
                self.params.model.train._n_epochs,
                self.params.model.train._n_train_samples_per_epoch,
                self.params.model.train.batch_size,
                self.params.model.train._n_val_samples_per_epoch,
                self.params.model.train.val_batch_size,
                __gen_queue_size__,
                __n_workers__,
                __n_gpus__,
                ' (CPU Merge: {}; CPU Reloc: {})'.format(__cpu_merge__, __cpu_reloc__)
                    if __n_gpus__ > 1
                    else '',
                filter))

        feature_cols = self.params.data._cat_prep_cols + self.params.data._num_prep_cols

        piece_paths = list(adf.piecePaths)
        random.shuffle(piece_paths)
        split_idx = int(math.ceil(self.params.model.train.train_proportion * adf.nPieces))
        train_piece_paths = piece_paths[:split_idx]
        val_piece_paths = piece_paths[split_idx:]

        if isinstance(model, BlueprintedArimoDLModel):
            assert isinstance(adf, S3ParquetDataFeeder)

            model.stdout_logger.info(model.config)

            n_threads = psutil.cpu_count(logical=True) - 2

            model.train_with_queue_reader_inputs(
                train_input=
                    adf._CrossSectDLDF(
                        feature_cols,
                        self.params.data.label.var,
                        piecePaths=train_piece_paths,
                        n=self.params.model.train.batch_size,
                        filter=filter,
                        nThreads=n_threads,
                        isRegression=True),

                val_input=
                    adf._CrossSectDLDF(
                        feature_cols,
                        self.params.data.label.var,
                        piecePaths=val_piece_paths,
                        n=self.params.model.train.val_batch_size,
                        filter=filter,
                        nThreads=n_threads,
                        isRegression=True),

                lr_scheduler=
                    LossPlateauLrDecay(
                        learning_rate=model.config.learning_rate,
                        decay_rate=model.config.lr_decay,
                        patience=self.params.model.train.reduce_lr_on_plateau.patience_n_epochs),

                max_epoch=self.params.model.train._n_epochs,

                early_stopping_patience=
                    max(self.params.model.train.early_stop.patience_min_n_epochs,
                        int(math.ceil(self.params.model.train.early_stop.patience_proportion_total_n_epochs *
                                      self.params.model.train._n_epochs))),

                num_train_batches_per_epoch=self.params.model.train._n_train_batches_per_epoch,

                num_test_batches_per_epoch=self.params.model.train._n_val_batches_per_epoch)

        else:
            assert isinstance(model, BlueprintedKerasModel)

            assert isinstance(adf, (S3ParquetDataFeeder, S3ParquetDistributedDataFrame))

            model.summary()

            if __n_gpus__ > 1:
                model._obj = \
                    arimo.backend.keras.utils.multi_gpu_model(
                        model._obj,
                        gpus=__n_gpus__,
                        cpu_merge=__cpu_merge__,
                        cpu_relocation=__cpu_reloc__)

            model.compile(
                loss=self.params.model.train.objective
                    if self.params.model.train.objective
                    else 'MAE',   # mae / mean_absolute_error (more resilient to outliers)
                optimizer=arimo.backend.keras.optimizers.Nadam(),
                metrics=[# 'MSE',   # mean_squared_error,
                         # 'MAPE'   # mean_absolute_percentage_error
                ])

            model.fit_generator(
                generator=adf.gen(
                            feature_cols,
                            self.params.data.label.var,
                            piecePaths=train_piece_paths,
                            n=__n_gpus__ * self.params.model.train.batch_size,
                            withReplacement=False,
                            seed=None,
                            anon=True,
                            collect='numpy',
                            pad=None,
                            cache=False,
                            filter=filter)(),
                steps_per_epoch=self.params.model.train._n_train_batches_per_epoch,
                epochs=self.params.model.train._n_epochs,
                verbose=2,
                callbacks=[
                    arimo.backend.keras.callbacks.TerminateOnNaN(),
                    arimo.backend.keras.callbacks.ReduceLROnPlateau(
                        monitor=self.params.model.train.val_metric.name,
                        factor=self.params.model.train.reduce_lr_on_plateau.factor,
                        patience=self.params.model.train.reduce_lr_on_plateau.patience_n_epochs,
                        verbose=int(verbose > 0),
                        mode=self.params.model.train.val_metric.mode,
                        min_delta=self.params.model.train.val_metric.significance,
                        cooldown=0,
                        min_lr=0),
                    arimo.backend.keras.callbacks.EarlyStopping(
                        monitor=self.params.model.train.val_metric.name,
                        min_delta=self.params.model.train.val_metric.significance,
                        patience=max(self.params.model.train.early_stop.patience_min_n_epochs,
                                     int(math.ceil(self.params.model.train.early_stop.patience_proportion_total_n_epochs *
                                                   self.params.model.train._n_epochs))),
                        verbose=int(verbose),
                        mode=self.params.model.train.val_metric.mode,
                        baseline=None)],
                validation_data=adf.gen(
                                    feature_cols,
                                    self.params.data.label.var,
                                    piecePaths=val_piece_paths,
                                    n=__n_gpus__ * self.params.model.train.val_batch_size,
                                    withReplacement=False,
                                    seed=None,
                                    anon=True,
                                    collect='numpy',
                                    pad=None,
                                    cache=False,
                                    filter=filter)(),
                validation_steps=self.params.model.train._n_val_batches_per_epoch,
                class_weight={},
                max_queue_size=__gen_queue_size__,
                workers=__n_workers__,
                use_multiprocessing=True,
                shuffle=False,
                initial_epoch=0)

        model.save()

        return model
