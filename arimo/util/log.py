import logging
import sys


# BUG FIX: https://github.com/tensorflow/tensorflow/issues/26691
try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


# handler for logging to StdOut
STDOUT_HANDLER = \
    logging.StreamHandler(sys.stdout)

STDOUT_HANDLER.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s   %(levelname)s   %(name)s:   %(message)s\n',
        datefmt='%Y-%m-%d %H:%M'))


# utility class to flush logging stream upon each write
# http://stackoverflow.com/questions/29772158/make-ipython-notebook-print-in-real-time
class _FlushFile:
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return self.f.__getattribute__(item)

    def flush(self):
        self.f.flush()

    def write(self, x):
        self.f.write(x)
        self.flush()


# enable live printing
def enable_live_print():
    sys.stdout = _FlushFile(sys.stdout)
    print('Live Printing Enabled')
