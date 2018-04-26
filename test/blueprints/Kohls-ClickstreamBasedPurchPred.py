import os

from __init__ import AI_PROJECTS_REPO_DIR_PATH


os.system(
    'python {0}/KohlsPurchPred/train.py --debug'.format(
        AI_PROJECTS_REPO_DIR_PATH))
