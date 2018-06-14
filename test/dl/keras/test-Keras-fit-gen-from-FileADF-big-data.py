from __future__ import division, print_function

import time
import tqdm

from arimo.blueprints import _blueprint_from_params
from arimo.util.pkl import pickle_able

from PanasonicColdChain.PredMaint.PPPAD import PRED_MAINT_PROJECT


DATASET_NAME = 'DISP_CASE---ex_display_case'


PPP_BP_UUID = 'DISP_CASE---ex_display_case---to-2017-03---07cf3d0e-6f9e-481f-8ed1-f68e3fbddce1'
LABEL_VAR = 'inside_temperature'


N_HID_NODES = 3 * (168,)

BATCH_SIZE = 10 ** 3
N_BATCHES = 10 ** 3
N_EPOCHS = 3


fadf = PRED_MAINT_PROJECT.load_equipment_data(DATASET_NAME, iCol=None)


ppp_bp = PRED_MAINT_PROJECT._ppp_blueprint(uuid=PPP_BP_UUID, verbose=False)
ppp_bp.params.model.ver = ppp_bp.model(ver='latest').ver


prep_fadf = ppp_bp.prep_data(df=fadf, mode='train')


component_bp = _blueprint_from_params(blueprint_params=ppp_bp.params.model.component_blueprints[LABEL_VAR])


gen = prep_fadf.gen(
    component_bp.params.data._cat_prep_cols + component_bp.params.data._num_prep_cols,
    LABEL_VAR,
    n=BATCH_SIZE,   # component_bp_params.model.train.batch_size,
    withReplacement=False,
    seed=None,
    anon=True,
    collect='numpy',
    pad=None,
    cache=False,
    filter={LABEL_VAR: (component_bp.params.data.label.lower_outlier_threshold,
                        component_bp.params.data.label.upper_outlier_threshold)})

pickle_able(gen)

g = gen()

for i in tqdm.tqdm(range(N_BATCHES)):
    _ = next(g)


component_bp.params.model.factory.n_hid_nodes = N_HID_NODES

model = component_bp.model(ver=None)

tic = time.time()

history = \
    model.fit_generator(
        generator=g,
        steps_per_epoch=N_BATCHES,
        epochs=N_EPOCHS,
        verbose=1,   # 1 = progress bar, 2 = one line per epoch.
        callbacks=None,
        validation_data=None,
        validation_steps=None,
        class_weight=None,

        max_queue_size=99,
        workers=3,
        use_multiprocessing=True,

        shuffle=False,
        initial_epoch=0) \
    .history

elapsed = time.time() - tic

n_mil_samples = N_EPOCHS * N_BATCHES * BATCH_SIZE / 1e6

print('TIME: {:,.0f}s for {:,.0f} mil samples = {:,.0f}s per mil samples'
      .format(elapsed, n_mil_samples, elapsed / n_mil_samples))
