import os
import time

from ..iterables import to_iterable
from . import client as aws_client


_AWS_ACCESS_KEY_ID_ENV_VAR_NAME = 'AWS_ACCESS_KEY_ID'
_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME = 'AWS_SECRET_ACCESS_KEY'


def client(access_key_id=None, secret_access_key=None):
    return aws_client(
            service='s3',
            access_key_id=access_key_id,
            secret_access_key=secret_access_key)


def s3a_path_with_auth(
        s3_path,   # TODO: /,
        *, access_key_id=None, secret_access_key=None):
    return 's3a://{}{}'.format(
            '{}:{}@'.format(
                access_key_id,
                secret_access_key)
            if access_key_id and secret_access_key
            else '',
            s3_path.split('://')[1])


def cp(from_path, to_path, is_dir=True,
       quiet=True,
       access_key_id=None, secret_access_key=None,
       verbose=True):
    if access_key_id and secret_access_key:
        _ORIG_AWS_ACCESS_KEY_ID = os.environ.get(_AWS_ACCESS_KEY_ID_ENV_VAR_NAME)
        _ORIG_AWS_SECRET_ACCESS_KEY = os.environ.get(_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME)

        os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = access_key_id
        os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = secret_access_key

    if verbose:
        msg = 'Copying "{}" to "{}"...'.format(from_path, to_path)
        print(msg + '\n')
        tic = time.time()

    os.system(
        'aws s3 cp {} {} {} {}'.format(
            from_path, to_path,
            '--recursive' if is_dir else '',
            '--quiet' if quiet else ''))

    if verbose:
        toc = time.time()
        print(msg + ' done!   <{:,.1f} s>\n'.format(toc - tic))

    if access_key_id and secret_access_key:
        if _ORIG_AWS_ACCESS_KEY_ID:
            os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = _ORIG_AWS_ACCESS_KEY_ID
        else:
            del os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME]

        if _ORIG_AWS_SECRET_ACCESS_KEY:
            os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = _ORIG_AWS_SECRET_ACCESS_KEY
        else:
            del os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME]


def mv(from_path, to_path, is_dir=True,
       quiet=True,
       access_key_id=None, secret_access_key=None,
       verbose=True):
    if access_key_id and secret_access_key:
        _ORIG_AWS_ACCESS_KEY_ID = os.environ.get(_AWS_ACCESS_KEY_ID_ENV_VAR_NAME)
        _ORIG_AWS_SECRET_ACCESS_KEY = os.environ.get(_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME)

        os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = access_key_id
        os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = secret_access_key

    if verbose:
        msg = 'Moving "{}" to "{}"...'.format(from_path, to_path)
        print(msg + '\n')
        tic = time.time()

    os.system(
        'aws s3 mv {} {} {} {}'.format(
            from_path, to_path,
            '--recursive' if is_dir else '',
            '--quiet' if quiet else ''))

    if verbose:
        toc = time.time()
        print(msg + ' done!   <{:,.1f} s>\n'.format(toc - tic))

    if access_key_id and secret_access_key:
        if _ORIG_AWS_ACCESS_KEY_ID:
            os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = _ORIG_AWS_ACCESS_KEY_ID
        else:
            del os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME]

        if _ORIG_AWS_SECRET_ACCESS_KEY:
            os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = _ORIG_AWS_SECRET_ACCESS_KEY
        else:
            del os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME]


def rm(path, dir=True, globs=None, quiet=True,
       access_key_id=None, secret_access_key=None,
       verbose=True):
    if access_key_id and secret_access_key:
        _ORIG_AWS_ACCESS_KEY_ID = os.environ.get(_AWS_ACCESS_KEY_ID_ENV_VAR_NAME)
        _ORIG_AWS_SECRET_ACCESS_KEY = os.environ.get(_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME)

        os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = access_key_id
        os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = secret_access_key

    if verbose:
        msg = 'Deleting {}"{}"...'.format(
                ('Globs "{}" @ '.format(globs)
                 if globs
                 else 'Directory ')
                    if dir
                    else '',
                path)

        print(msg)

    os.system(
        'aws s3 rm {}{}{}'.format(
            path,
            ' --recursive{}'.format(
                    ' --exclude "*" {}'.format(
                        ' '.join('--include "{}"'.format(glob)
                                 for glob in to_iterable(globs)))
                    if globs
                    else '')
                if dir
                else '',
            ' --quiet' if quiet else ''))

    if verbose:
        print(msg + ' done!')

    if access_key_id and secret_access_key:
        if _ORIG_AWS_ACCESS_KEY_ID:
            os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = _ORIG_AWS_ACCESS_KEY_ID
        else:
            del os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME]

        if _ORIG_AWS_SECRET_ACCESS_KEY:
            os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = _ORIG_AWS_SECRET_ACCESS_KEY
        else:
            del os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME]


def sync(from_dir_path, to_dir_path,
         delete=True, quiet=True,
         access_key_id=None, secret_access_key=None,
         verbose=True):
    if access_key_id and secret_access_key:
        _ORIG_AWS_ACCESS_KEY_ID = os.environ.get(_AWS_ACCESS_KEY_ID_ENV_VAR_NAME)
        _ORIG_AWS_SECRET_ACCESS_KEY = os.environ.get(_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME)

        os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = access_key_id
        os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = secret_access_key

    if verbose:
        msg = 'Syncing "{}" to "{}"...'.format(from_dir_path, to_dir_path)
        print(msg + '\n')
        tic = time.time()

    os.system(
        'aws s3 sync {} {} {} {}'.format(
            from_dir_path, to_dir_path,
            '--delete' if delete else '',
            '--quiet' if quiet else ''))

    if verbose:
        toc = time.time()
        print(msg + ' done!   <{:,.1f} s>\n'.format(toc - tic))

    if access_key_id and secret_access_key:
        if _ORIG_AWS_ACCESS_KEY_ID:
            os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME] = _ORIG_AWS_ACCESS_KEY_ID
        else:
            del os.environ[_AWS_ACCESS_KEY_ID_ENV_VAR_NAME]

        if _ORIG_AWS_SECRET_ACCESS_KEY:
            os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME] = _ORIG_AWS_SECRET_ACCESS_KEY
        else:
            del os.environ[_AWS_SECRET_ACCESS_KEY_ENV_VAR_NAME]
