"""Logging-related utilities."""


import logging
import sys


__all__ = 'STDOUT_HANDLER', 'enable_live_print'


# handler for logging to StdOut
STDOUT_HANDLER = logging.StreamHandler(stream=sys.stdout)

STDOUT_HANDLER.setFormatter(
    fmt=logging.Formatter(
        fmt='%(asctime)s   %(levelname)s   %(name)s:   %(message)s\n',
        datefmt='%Y-%m-%d %H:%M',
        style='%',
        validate=True))


# utility class to flush logging stream upon each write
# stackoverflow.com/questions/29772158/make-ipython-notebook-print-in-real-time
class _FlushFile:
    def __init__(self, f):   # pylint: disable=invalid-name
        self.f = f   # pylint: disable=invalid-name

    def __getattr__(self, item):
        return self.f.__getattribute__(item)

    def flush(self):
        """Flush."""
        self.f.flush()

    def write(self, x):   # pylint: disable=invalid-name
        """Write."""
        self.f.write(x)
        self.flush()


def enable_live_print():
    """Enable live printing."""
    sys.stdout = _FlushFile(sys.stdout)
    print('Live Printing Enabled')
