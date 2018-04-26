import os

import arimo


AI_PROJECTS_REPO_DIR_PATH = \
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                arimo.__path__[0])),
        'AI-projects')


PANCC_REPO_DIR_PATH = \
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                arimo.__path__[0])),
        'PanasonicColdChain-projects')
