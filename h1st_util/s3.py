"""AWS S3 utilities."""


import os
import time
from typing import Optional

from .fs import PathType
from .iter import to_iterable


__all__ = 'cp', 'mv', 'rm', 'sync'


def cp(from_path: PathType, to_path: PathType,
       *, is_dir: bool = True,
       quiet: bool = True, verbose: bool = True):
    # pylint: disable=invalid-name,too-many-arguments
    """S3 Copy."""
    s3_command: str = (f'aws s3 cp {from_path} {to_path}' +
                       (' --recursive' if is_dir else '') +
                       (' --quiet' if quiet else ''))

    if verbose:
        msg: str = f'Copying "{from_path}" to "{to_path}"...'
        print(msg)
        print(f'Running: {s3_command}...')
        tic: float = time.time()

    os.system(command=s3_command)

    if verbose:
        toc: float = time.time()
        print(f'{msg} done!   <{(toc - tic):,.1f} s>')


def mv(from_path: PathType, to_path: PathType,
       *, is_dir: bool = True,
       quiet: bool = True, verbose: bool = True):
    # pylint: disable=invalid-name,too-many-arguments
    """S3 Move."""
    s3_command: str = (f'aws s3 mv {from_path} {to_path}' +
                       (' --recursive' if is_dir else '') +
                       (' --quiet' if quiet else ''))
    if verbose:
        msg: str = f'Moving "{from_path}" to "{to_path}"...'
        print(msg)
        print(f'Running: {s3_command}...')
        tic: float = time.time()

    os.system(command=s3_command)

    if verbose:
        toc: float = time.time()
        print(f'{msg} done!   <{(toc - tic):,.1f} s>')


def rm(path: PathType,
       *, is_dir: bool = True, globs: Optional[str] = None,
       quiet: bool = True, verbose: bool = True):
    # pylint: disable=invalid-name,too-many-arguments
    """S3 Remove."""
    s3_command: str = (f'aws s3 rm {path}' +
                       ((' --recursive' +
                         ((' --exclude "*" ' +
                           ' '.join(f'--include "{glob}"'
                                    for glob in to_iterable(globs)))
                          if globs
                          else ''))
                        if is_dir
                        else '') +
                       (' --quiet' if quiet else ''))

    if verbose:
        msg: str = ('Deleting ' +
                    ((f'Globs "{globs}" @ ' if globs else 'Directory ')
                     if is_dir
                     else '') +
                    f'"{path}"...')
        print(msg)
        print(f'Running: {s3_command}...')
        tic: float = time.time()

    os.system(command=s3_command)

    if verbose:
        toc: float = time.time()
        print(f'{msg} done!   <{(toc - tic):,.1f} s>')


def sync(from_dir_path: PathType, to_dir_path: PathType,
         *, delete: bool = True,
         quiet: bool = True, verbose=True):
    # pylint: disable=too-many-arguments
    """S3 Sync."""
    s3_command: str = (f'aws s3 sync {from_dir_path} {to_dir_path}' +
                       (' --delete' if delete else '') +
                       (' --quiet' if quiet else ''))
    if verbose:
        msg: str = f'Syncing "{from_dir_path}" to "{to_dir_path}"...'
        print(msg)
        print(f'Running: {s3_command}...')
        tic = time.time()

    os.system(command=s3_command)

    if verbose:
        toc = time.time()
        print(f'{msg} done!   <{(toc - tic):,.1f} s>')
