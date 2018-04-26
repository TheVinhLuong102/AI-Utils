import argparse
import os

from __init__ import AI_PROJECTS_REPO_DIR_PATH


_arg_parser = \
    argparse.ArgumentParser(
        argument_default=None)

_arg_parser.add_argument(
    '--debug',
    action='store_true')

_arg_parser.add_argument(
    '--orig-keras',
    action='store_true')

_arg_parser.add_argument(
    'equipment_data_field_group_monitor_names',
    nargs='*',
    default=[])

_args = _arg_parser.parse_args()


os.system(
    'python {0}/Caterpillar/Anomaly/train.py {1} --debug {2}'.format(
        AI_PROJECTS_REPO_DIR_PATH,
        ' '.join(_args.equipment_data_field_group_monitor_names),
        '--orig-keras'
            if _args.orig_keras
            else ''))
