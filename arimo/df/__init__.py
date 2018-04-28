import abc
import logging

import six
_STR_CLASSES = \
    (str, unicode) \
    if six.PY2 \
    else str

from arimo.util.date_time import \
    (DATE_COL,
     _T_ORD_COL, _T_DELTA_COL,
     _T_HoY_COL, _T_QoY_COL, _T_MoY_COL, _T_PoY_COL,   # _T_WoY_COL, _T_DoY_COL,
     _T_QoH_COL, _T_MoH_COL, _T_PoH_COL,
     _T_MoQ_COL, _T_PoQ_COL,
     _T_WoM_COL, _T_DoM_COL, _T_PoM_COL,
     _T_DoW_COL, _T_PoW_COL,
     _T_HoD_COL, _T_PoD_COL,
     _T_COMPONENT_AUX_COLS, _T_CAT_AUX_COLS, _T_NUM_AUX_COLS)
from arimo.util.log import STDOUT_HANDLER
import arimo.debug


class _DF_ABC(object):
    __metaclass__ = abc.ABCMeta

    # default identity/entity, timestamp & time order columns
    _DEFAULT_I_COL = 'id'

    _DEFAULT_T_COL = 't'
    _DEFAULT_D_COL = DATE_COL

    _T_ORD_COL = _T_ORD_COL

    _T_DELTA_COL = _T_DELTA_COL

    _T_REL_AUX_COLS = _T_ORD_COL, _T_DELTA_COL

    _T_HoY_COL = _T_HoY_COL   # Half of Year
    _T_QoY_COL = _T_QoY_COL   # Quarter of Year
    _T_MoY_COL = _T_MoY_COL   # Month of Year
    # _T_WoY_COL = _T_WoY_COL   # Week of Year
    # _T_DoY_COL = _T_DoY_COL   # Day of Year
    _T_PoY_COL = _T_PoY_COL   # Part/Proportion/Fraction of Year

    _T_QoH_COL = _T_QoH_COL   # Quarter of Half-Year
    _T_MoH_COL = _T_MoH_COL   # Month of Half-Year
    _T_PoH_COL = _T_PoH_COL   # Part/Proportion/Fraction of Half-Year

    _T_MoQ_COL = _T_MoQ_COL   # Month of Quarter
    _T_PoQ_COL = _T_PoQ_COL   # Part/Proportion/Fraction of Quarter

    _T_WoM_COL = _T_WoM_COL   # Week of Month
    _T_DoM_COL = _T_DoM_COL   # Day of Month
    _T_PoM_COL = _T_PoM_COL   # Part/Proportion/Fraction of Month

    _T_DoW_COL = _T_DoW_COL   # Day of Week
    _T_PoW_COL = _T_PoW_COL   # Part/Proportion/Fraction of Week

    _T_HoD_COL = _T_HoD_COL   # Hour of Day
    _T_PoD_COL = _T_PoD_COL   # Part/Proportion/Fraction of Day

    _T_COMPONENT_AUX_COLS = _T_COMPONENT_AUX_COLS

    _T_AUX_COLS = _T_REL_AUX_COLS + _T_COMPONENT_AUX_COLS

    _T_CAT_AUX_COLS = _T_CAT_AUX_COLS

    _T_NUM_AUX_COLS = _T_NUM_AUX_COLS

    # repr sample size
    _DEFAULT_REPR_SAMPLE_SIZE = 10 ** 6

    # prep col prefixes / suffix
    _NULL_FILL_PREFIX = '__NullFill__'

    _CAT_IDX_PREFIX = '__CatIdx__'
    _OHE_PREFIX = '__OHE__'

    _STD_SCL_PREFIX = '__StdScl__'
    _MAX_ABS_SCL_PREFIX = '__MaxAbsScl__'

    _MIN_MAX_SCL_PREFIX = '__MinMaxScl__'

    _PREP_SUFFIX = '__'

    _CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME = 'catOrigToPrepColMap.json'
    _NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME = 'numOrigToPrepColMap.json'
    _VEC_COL_MAP_FILE_NAME = 'vecColMap.json'

    # temp dir
    _TMP_DIR_PATH = '/tmp/.arimo/df'

    # data prep cache
    _PREP_CACHE = {}

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractproperty
    def __short_repr__(self):
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def __unicode__(self):
        return repr(self)

    @classmethod
    def class_logger(cls, *handlers, **kwargs):
        logger = logging.getLogger(name='{}'.format(cls.__name__))

        level = kwargs.get('level')

        if level is None:
            level = logging.DEBUG \
                if arimo.debug.ON \
                else logging.INFO

        logger.setLevel(level)

        if kwargs.get('verbose'):
            handlers += (STDOUT_HANDLER,)

        for handler in handlers:
            logger.addHandler(handler)

        return logger

    @classmethod
    def class_stdout_logger(cls):
        return cls.class_logger(
            level=logging.DEBUG,
            verbose=True)

    def logger(self, *handlers, **kwargs):
        logger = logging.getLogger(name=self.__short_repr__)

        level = kwargs.get('level')

        if level is None:
            level = logging.DEBUG \
                if arimo.debug.ON \
                else logging.INFO

        logger.setLevel(level)

        if kwargs.get('verbose'):
            handlers += (STDOUT_HANDLER,)

        for handler in handlers:
            logger.addHandler(handler)

        return logger

    @property
    def stdout_logger(self):
        return self.logger(
            level=logging.DEBUG,
            verbose=True)
