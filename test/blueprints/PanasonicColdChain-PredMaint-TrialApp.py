import os

from __init__ import PANCC_REPO_DIR_PATH


os.system(
    'python {}/PanasonicColdChain/PredMaint/PENG_Apps/zzzTrial/train.py --debug'
        .format(PANCC_REPO_DIR_PATH))
