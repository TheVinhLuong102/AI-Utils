import os
from pyarrow.hdfs import HadoopFileSystem
import shutil
import subprocess
import sys


# Hadoop configuration directory
_HADOOP_HOME_ENV_VAR_NAME = 'HADOOP_HOME'

_HADOOP_HOME = os.environ.get(_HADOOP_HOME_ENV_VAR_NAME)


def _hdfs_cmd(hadoop_home=_HADOOP_HOME):
    if hadoop_home:
        cmd = '{0}/bin/hdfs'.format(hadoop_home)

        if not os.path.isfile(cmd):
            cmd = 'hdfs'

    else:
        cmd = 'hdfs'

    return cmd


_HADOOP_CONF_DIR_ENV_VAR_NAME = 'HADOOP_CONF_DIR'


# check if running on Linux cluster or local Mac
_ON_LINUX_CLUSTER = sys.platform.startswith('linux')

# detect & set up HDFS client
if _HADOOP_HOME:
    os.environ['ARROW_LIBHDFS_DIR'] = \
        os.path.join(
            _HADOOP_HOME,
            'lib',
            'native')

    try:
        hdfs_client = HadoopFileSystem()

        try:
            print('Testing HDFS... ', end='')

            if hdfs_client.isdir('/'):
                _ON_LINUX_CLUSTER_WITH_HDFS = True
                print('done!')

            else:
                _ON_LINUX_CLUSTER_WITH_HDFS = False
                print('UNAVAILABLE')

        except:
            hdfs_client = None
            _ON_LINUX_CLUSTER_WITH_HDFS = False
            print('UNAVAILABLE')

    except:
        hdfs_client = None
        _ON_LINUX_CLUSTER_WITH_HDFS = False
        print('*** HDFS UNAVAILABLE ***')

else:
    hdfs_client = None
    _ON_LINUX_CLUSTER_WITH_HDFS = False
    print('*** HDFS UNAVAILABLE ***')


def _exec(cmd, must_succeed=False):
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    out, err = p.communicate()

    if must_succeed and p.returncode:
        raise RuntimeError(
            '*** COMMAND ERROR: {0} ***\n{1}\n{2}\n'
                .format(cmd, out, err))


def command_prefix(hdfs=True, hadoop_home='/opt/hadoop'):
    return '{0} dfs -'.format(_hdfs_cmd(hadoop_home=hadoop_home)) \
        if hdfs \
        else ''


def exist(path, hdfs=False, dir=False):
    return (hdfs_client.isdir(path=path)
             if dir
             else hdfs_client.isfile(path=path)) \
        if hdfs and _ON_LINUX_CLUSTER_WITH_HDFS \
        else (os.path.isdir(path)
              if dir
              else (os.path.isfile(path) or os.path.islink(path)))


def mkdir(dir, hdfs=True, hadoop_home='/opt/hadoop'):
    cmd_pref = command_prefix(hdfs=hdfs, hadoop_home=hadoop_home)

    cmd = '{}mkdir -p "{}"{}'.format(
        cmd_pref,
        dir,
        ' -m 0777'
            if _ON_LINUX_CLUSTER and (not hdfs)
            else '')

    _ = os.system(cmd)
    assert _ <= 0, 'FAILED: {} (EXIT CODE: {})'.format(cmd, _)


def rm(path, hdfs=True, is_dir=True, hadoop_home='/opt/hadoop'):
    if not _ON_LINUX_CLUSTER_WITH_HDFS:
        hdfs = False

    if hdfs:
        os.system('{}rm{} -skipTrash "{}"'.format(
            command_prefix(hdfs=True, hadoop_home=hadoop_home),
            ' -r' if is_dir else '',
            path))

    elif is_dir and os.path.isdir(path):
        try:
            shutil.rmtree(
                path=path,
                ignore_errors=False)

        except:
            os.system('rm -f "{}"'.format(path))

        assert not os.path.isdir(path), \
            '*** CANNOT REMOVE LOCAL DIR "{}" ***'.format(path)

    elif os.path.isfile(path) or os.path.islink(path):
        os.remove(path)

        assert not (os.path.isfile(path) or os.path.islink(path)), \
            '*** CANNOT REMOVE LOCAL FILE/SYMLINK "{}" ***'.format(path)


def empty(dir, hdfs=True, hadoop_home='/opt/hadoop'):
    if exist(path=dir, hdfs=hdfs, dir=True):
        rm(path=dir, hdfs=hdfs, is_dir=True, hadoop_home=hadoop_home)
    mkdir(dir=dir, hdfs=hdfs, hadoop_home=hadoop_home)


def cp(from_path, to_path, hdfs=True, is_dir=True, hadoop_home='/opt/hadoop'):
    rm(path=to_path, hdfs=hdfs, is_dir=is_dir, hadoop_home=hadoop_home)

    par_dir_path = os.path.dirname(to_path)
    if par_dir_path:
        mkdir(dir=par_dir_path, hdfs=hdfs, hadoop_home=hadoop_home)

    if hdfs:
        os.system('{}cp "{}" "{}"'.format(
            command_prefix(hdfs=True, hadoop_home=hadoop_home),
            from_path, to_path))

    elif is_dir:
        shutil.copytree(
            src=from_path,
            dst=to_path,
            symlinks=False,
            ignore=None)

    else:
        shutil.copyfile(
            src=from_path,
            dst=to_path)


def mv(from_path, to_path, hdfs=True, is_dir=True, hadoop_home='/opt/hadoop'):
    rm(path=to_path, hdfs=hdfs, is_dir=is_dir, hadoop_home=hadoop_home)

    par_dir_path = os.path.dirname(to_path)
    if par_dir_path:
        mkdir(dir=par_dir_path, hdfs=hdfs, hadoop_home=hadoop_home)

    if hdfs:
        os.system('{}mv "{}" "{}"'.format(
            command_prefix(hdfs=hdfs, hadoop_home=hadoop_home),
            from_path, to_path))

    else:
        try:
            shutil.move(
                src=from_path,
                dst=to_path)

        except:
            os.system('mv "{}" "{}"'.format(
                from_path, to_path))


def get(from_hdfs, to_local,
        is_dir=False, overwrite=True, _mv=False,
        hadoop_home='/opt/hadoop',
        must_succeed=False,
        _on_linux_cluster_with_hdfs=_ON_LINUX_CLUSTER_WITH_HDFS):
    if _on_linux_cluster_with_hdfs:
        if overwrite:
            rm(path=to_local,
               hdfs=False,
               is_dir=is_dir)

        if overwrite or \
                (is_dir and (not os.path.isdir(to_local))) or \
                ((not is_dir) and (not os.path.isfile(to_local))):
            par_dir_path = os.path.dirname(to_local)
            if par_dir_path:
                mkdir(
                    dir=par_dir_path,
                    hdfs=False)

            cmd = '{0} dfs -get "{1}" "{2}"'.format(
                    _hdfs_cmd(hadoop_home=hadoop_home),
                    from_hdfs, to_local)
            _exec(cmd)

            if _mv:
                rm(path=from_hdfs,
                   hdfs=True,
                   is_dir=is_dir,
                   hadoop_home=hadoop_home)

    elif from_hdfs != to_local:
        if _mv:
            mv(from_path=from_hdfs,
               to_path=to_local,
               hdfs=False,
               is_dir=is_dir)

        else:
            cp(from_path=from_hdfs,
               to_path=to_local,
               hdfs=False,
               is_dir=is_dir)

    if must_succeed:
        assert os.path.isdir(to_local) \
            if is_dir \
          else os.path.isfile(to_local), \
            '*** FS.GET({} -> {}) FAILED! ***'.format(from_hdfs, to_local)


def put(from_local, to_hdfs,
        is_dir=True, _mv=True, hadoop_home='/opt/hadoop'):
    if _ON_LINUX_CLUSTER_WITH_HDFS:
        rm(path=to_hdfs,
           hdfs=True,
           is_dir=is_dir,
           hadoop_home=hadoop_home)

        par_dir_path = os.path.dirname(to_hdfs)
        if par_dir_path:
            mkdir(
                dir=par_dir_path,
                hdfs=True,
                hadoop_home=hadoop_home)

        os.system(
            '{0} dfs -put "{1}" "{2}"'.format(
                _hdfs_cmd(hadoop_home=hadoop_home),
                from_local, to_hdfs))

        if _mv:
            rm(path=from_local,
               hdfs=False,
               is_dir=is_dir)

    elif from_local != to_hdfs:
        if _mv:
            mv(from_path=from_local,
               to_path=to_hdfs,
               hdfs=False,
               is_dir=is_dir)

        else:
            cp(from_path=from_local,
               to_path=to_hdfs,
               hdfs=False,
               is_dir=is_dir)
