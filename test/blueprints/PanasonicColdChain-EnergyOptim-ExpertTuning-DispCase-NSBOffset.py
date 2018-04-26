import os

from __init__ import PANCC_REPO_DIR_PATH


os.system(
    'python {}/PanasonicColdChain/EnergyOptim/PENG-Apps/ExpertTuning/DispCase/NSBOffset/train.py --debug'
        .format(PANCC_REPO_DIR_PATH))
