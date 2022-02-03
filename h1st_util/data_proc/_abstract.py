"""Abstract Data Handlers."""


from __future__ import annotations

import logging
from pathlib import Path
import tempfile
from typing import Any, Optional, Union
from typing import Collection, Dict, Set, Tuple   # Py3.9+: built-ins

from numpy import ndarray
from pandas import DataFrame, Series

from .. import debug, s3
from ..log import STDOUT_HANDLER
from ..namespace import Namespace


__all__ = (
    'AbstractDataHandler',
    'AbstractFileDataHandler',
    'AbstractS3FileDataHandler',
    'ReducedDataSetType',
)


ReducedDataSetType = Union[Any, Collection, ndarray, DataFrame, Series]


class AbstractDataHandler:
    # pylint: disable=no-member,too-many-public-methods
    """Abstract Data Handler."""

    # default identity/entity, timestamp & date columns
    _DEFAULT_I_COL: str = 'id'
    _DEFAULT_T_COL: str = 't'
    _DEFAULT_D_COL: str = 'date'

    # default representative sample size
    _DEFAULT_REPR_SAMPLE_SIZE: int = 10 ** 6

    # default profiling settings
    _DEFAULT_MIN_NON_NULL_PROPORTION: float = .32
    _DEFAULT_OUTLIER_TAIL_PROPORTION: float = 1e-3   # 0.1% each tail
    _DEFAULT_MAX_N_CATS: int = 12   # MoY is likely most numerous-category var
    _DEFAULT_MIN_PROPORTION_BY_MAX_N_CATS: float = .9

    # prep column prefixes/suffixes
    _NULL_FILL_PREFIX: str = 'NULL_FILLED__'

    _CAT_IDX_PREFIX: str = 'CAT_INDEX__'

    _STD_SCL_PREFIX: str = 'STD_SCALED__'
    _MAX_ABS_SCL_PREFIX: str = 'MAXABS_SCALED__'
    _MIN_MAX_SCL_PREFIX: str = 'MINMAX_SCALED__'

    _CAT_ORIG_TO_PREP_COL_MAP_FILE_NAME: str = 'cat-orig-to-prep-col-map.yaml'
    _NUM_ORIG_TO_PREP_COL_MAP_FILE_NAME: str = 'num-orig-to-prep-col-map.yaml'

    # temp dir
    _TMP_DIR_PATH: Path = (Path(tempfile.gettempdir()).resolve(strict=True) /
                           '.h1st/tmp')

    # data prep cache
    _PREP_CACHE: Dict[Path, Namespace] = {}

    # ============
    # REPR METHODS
    # ------------
    # __repr__
    # __shortRepr__
    # __str__

    def __repr__(self) -> str:
        """Return string repr."""
        raise NotImplementedError

    @property
    def __shortRepr__(self) -> str:   # noqa: N802
        # pylint: disable=invalid-name
        """Return short string repr."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return string repr."""
        return repr(self)

    # =======
    # LOGGERS
    # -------
    # classLogger
    # classStdOutLogger
    # logger
    # stdOutLogger

    @classmethod
    def classLogger(cls,   # noqa: N802
                    *handlers: logging.Handler,
                    **kwargs: Any) -> logging.Logger:
        # pylint: disable=invalid-name
        """Get Class Logger."""
        logger: logging.Logger = logging.getLogger(name=cls.__name__)

        level: Optional[int] = kwargs.get('level')
        if not level:
            level: int = logging.DEBUG if debug.ON else logging.INFO
        logger.setLevel(level=level)

        for handler in handlers:
            logger.addHandler(hdlr=handler)
        if kwargs.get('verbose'):
            logger.addHandler(hdlr=STDOUT_HANDLER)

        return logger

    @classmethod
    def classStdOutLogger(cls) -> logging.Logger:   # noqa: N802
        # pylint: disable=invalid-name
        """Get Class StdOut Logger."""
        return cls.classLogger(level=logging.DEBUG, verbose=True)

    def logger(self, *handlers: logging.Handler, **kwargs: Any) -> logging.Logger:   # noqa: E501
        """Get Logger."""
        logger: logging.Logger = logging.getLogger(name=self.__shortRepr__)

        level: Optional[int] = kwargs.get('level')
        if not level:
            level: int = logging.DEBUG if debug.ON else logging.INFO
        logger.setLevel(level=level)

        for handler in handlers:
            logger.addHandler(hdlr=handler)
        if kwargs.get('verbose'):
            logger.addHandler(hdlr=STDOUT_HANDLER)

        return logger

    @property
    def stdOutLogger(self) -> logging.Logger:   # noqa: N802
        # pylint: disable=invalid-name
        """Get StdOut Logger."""
        return self.logger(level=logging.DEBUG, verbose=True)

    # ===============
    # CACHING METHODS
    # ---------------
    # _emptyCache
    # _inheritCache

    def _emptyCache(self):   # noqa: N802
        # pylint: disable=invalid-name
        """Empty cache."""
        raise NotImplementedError

    def _inheritCache(self, *args: Any, **kwargs: Any):   # noqa: N802
        # pylint: disable=invalid-name
        """Inherit existing cache."""
        raise NotImplementedError

    # =========================
    # KEY (SETTABLE) PROPERTIES
    # -------------------------
    # _assignReprSample
    # reprSampleSize
    # reprSample
    # minNonNullProportion
    # outlierTailProportion
    # maxNCats
    # minProportionByMaxNCats

    def _assignReprSample(self):   # noqa: N802
        # pylint: disable=invalid-name
        """Assign representative sample."""
        raise NotImplementedError

    @property
    def reprSampleSize(self) -> int:   # noqa: N802
        # pylint: disable=invalid-name
        """Return approx number of rows to sample for profiling purposes.

        (default = 1,000,000)
        """
        if self._cache.reprSample is None:
            self._assignReprSample()

        return self._reprSampleSize

    @reprSampleSize.setter
    def reprSampleSize(self, n: int, /):   # noqa: N802
        # pylint: disable=invalid-name
        self._reprSampleSize: int = n
        self._assignReprSample()

    @property
    def reprSample(self):   # noqa: N802
        # pylint: disable=invalid-name
        """Sub-sampled data set according to ``.reprSampleSize`` attribute."""
        if self._cache.reprSample is None:
            self._assignReprSample()

        return self._cache.reprSample

    @property
    def minNonNullProportion(self) -> float:   # noqa: N802
        # pylint: disable=invalid-name
        """Return min proportion of non-NULL values in each column.

        (to qualify it as a valid feature to use in downstream modeling)
        (default = .32)
        """
        return self._minNonNullProportion.default

    @minNonNullProportion.setter
    def minNonNullProportion(self, proportion: float, /):   # noqa: N802
        # pylint: disable=invalid-name
        if proportion != self._minNonNullProportion.default:
            self._minNonNullProportion.default = proportion
            self._cache.suffNonNull = {}

    @property
    def outlierTailProportion(self) -> float:   # noqa: N802
        # pylint: disable=invalid-name
        """Return proportion in each tail of each numerical column's distrib.

        (to exclude when computing outlier-resistant statistics)
        (default = .001)
        """
        return self._outlierTailProportion.default

    @outlierTailProportion.setter
    def outlierTailProportion(self, proportion: float, /):   # noqa: N802
        # pylint: disable=invalid-name
        self._outlierTailProportion.default = proportion

    @property
    def maxNCats(self) -> int:   # noqa: N802
        # pylint: disable=invalid-name
        """Return max number of categorical levels for possible cat. columns.

        (default = 12)
        """
        return self._maxNCats.default

    @maxNCats.setter
    def maxNCats(self, n: int, /):   # noqa: N802
        # pylint: disable=invalid-name
        self._maxNCats.default = n

    @property
    def minProportionByMaxNCats(self) -> float:   # noqa: N802
        # pylint: disable=invalid-name
        """Return min total proportion accounted for by top ``maxNCats``.

        (to consider the column truly categorical)
        (default = .9)
        """
        return self._minProportionByMaxNCats.default

    @minProportionByMaxNCats.setter
    def minProportionByMaxNCats(self, proportion: float, /):   # noqa: N802
        # pylint: disable=invalid-name
        self._minProportionByMaxNCats.default = proportion

    # =====================
    # ROWS, COLUMNS & TYPES
    # ---------------------
    # __len__ / nRows
    # types / type / typeIsNum

    def __len__(self) -> int:
        """Return number of rows."""
        return self.nRows

    @property
    def nRows(self) -> int:   # noqa: N802
        # pylint: disable=invalid-name
        """Return number of rows."""
        raise NotImplementedError

    @nRows.deleter
    def nRows(self):   # noqa: N802
        # pylint: disable=invalid-name
        self._cache.nRows = None

    @property
    def types(self) -> Namespace:
        """Return column data types."""
        raise NotImplementedError

    def type(self, col: str) -> type:
        """Return data type of specified column."""
        raise NotImplementedError

    def typeIsNum(self, col: str) -> bool:   # noqa: N802
        # pylint: disable=invalid-name
        """Check whether specified column's data type is numerical."""
        raise NotImplementedError

    # =============
    # COLUMN GROUPS
    # -------------
    # columns
    # indexCols
    # contentCols
    # possibleFeatureCols
    # possibleCatCols
    # possibleNumCols

    @property
    def columns(self) -> Set[str]:
        """Return columns."""
        raise NotImplementedError

    @property
    def indexCols(self) -> Set[str]:   # noqa: N802
        # pylint: disable=invalid-name
        """Return index columns."""
        raise NotImplementedError

    @property
    def contentCols(self) -> Set[str]:   # noqa: N802
        # pylint: disable=invalid-name
        """Return content columns."""
        return self.columns - self.indexCols

    @property
    def possibleFeatureCols(self) -> Set[str]:   # noqa: N802
        # pylint: disable=invalid-name
        """Return possible feature content columns."""
        raise NotImplementedError

    @property
    def possibleCatCols(self) -> Set[str]:   # noqa: N802
        # pylint: disable=invalid-name
        """Return possible categorical content columns."""
        raise NotImplementedError

    @property
    def possibleNumCols(self) -> Set[str]:   # noqa: N802
        # pylint: disable=invalid-name
        """Return possible numerical content columns."""
        return {col for col in self.contentCols if self.typeIsNum(col)}

    # ================
    # COLUMN PROFILING
    # ----------------
    # count
    # nonNullProportion
    # suffNonNull
    # distinct
    # quantile
    # sampleStat
    # outlierRstStat
    # profile

    def count(self, *cols: str, **kwargs: Any) -> Union[int, Namespace]:
        """Count non-NULL data values in specified column(s)."""
        raise NotImplementedError

    def nonNullProportion(self, *cols: str, **kwargs: Any) \
            -> Union[float, Namespace]:   # noqa: N802
        # pylint: disable=invalid-name
        """Count non-NULL data proportion(s) in specified column(s)."""
        raise NotImplementedError

    def suffNonNull(self, *cols: str, **kwargs: Any) -> Union[bool, Namespace]:   # noqa: E501,N802
        # pylint: disable=invalid-name:
        """Check whether columns have sufficient non-NULL values.

        (at least ``.minNonNullProportion`` of values being non-``NULL``)

        Return:
            - If 1 column name is given, return ``True``/``False``

            - If multiple column names are given,
            return a {``col``: ``True`` or ``False``} *dict*

            - If no column names are given,
            return a {``col``: ``True`` or ``False``} *dict* for all columns
        """
        if not cols:
            cols: Tuple[str] = tuple(self.contentCols)

        if len(cols) > 1:
            return Namespace(**{col: self.suffNonNull(col, **kwargs)
                                for col in cols})

        col: str = cols[0]

        minNonNullProportion: float = self._minNonNullProportion[col]   # noqa: E501,N806

        outdatedSuffNonNullProportionThreshold: bool = False   # noqa: N806

        if col in self._cache.suffNonNullProportionThreshold:
            if self._cache.suffNonNullProportionThreshold[col] != \
                    minNonNullProportion:
                outdatedSuffNonNullProportionThreshold: bool = True   # noqa: E501,N806
                self._cache.suffNonNullProportionThreshold[col] = \
                    minNonNullProportion

        else:
            self._cache.suffNonNullProportionThreshold[col] = \
                minNonNullProportion

        if (col not in self._cache.suffNonNull) or \
                outdatedSuffNonNullProportionThreshold:
            self._cache.suffNonNull[col] = (
                self.nonNullProportion(col) >=
                self._cache.suffNonNullProportionThreshold[col])

        return self._cache.suffNonNull[col]

    def distinct(self, *cols: str, **kwargs: Any) -> Union[Dict[str, float],
                                                           Series, Namespace]:
        """Return distinct values for specified column(s)."""
        raise NotImplementedError

    def quantile(self, *cols: str, **kwargs: Any) -> Union[float, int,
                                                           Series, Namespace]:
        """Return quantile values for specified column(s)."""
        raise NotImplementedError

    def sampleStat(self, *cols: str, **kwargs: Any) \
            -> Union[float, int, Namespace]:   # noqa: N802
        # pylint: disable=invalid-name:
        """Return certain sample statistics for specified columns."""
        raise NotImplementedError

    def outlierRstStat(self, *cols: str, **kwargs: Any) \
            -> Union[float, int, Namespace]:   # noqa: N802
        # pylint: disable=invalid-name:
        """Return outlier-resistant statistics for specified columns."""
        raise NotImplementedError

    def profile(self, *cols: str, **kwargs: Any) -> Namespace:
        """Profile specified column(s)."""
        raise NotImplementedError

    # =========
    # DATA PREP
    # ---------
    # fillNumNull
    # preprocessForML

    def fillNumNull(self, *cols: str, **kwargs: Any) -> AbstractDataHandler:   # noqa: E501,N802
        # pylint: disable=invalid-name
        """Fill NULL values for specified column(s)."""
        raise NotImplementedError

    def preprocessForML(self, *cols: str, **kwargs: Any) -> AbstractDataHandler:   # noqa: E501,N802
        # pylint: disable=invalid-name
        """Pre-process specified column(s) for ML model training/inference."""
        raise NotImplementedError

    # ========
    # SAMPLING
    # --------
    # sample

    def sample(self, *cols: str, **kwargs: Any) -> Union[ReducedDataSetType, Any]:   # noqa: E501
        """Sample from data set."""
        raise NotImplementedError


class AbstractFileDataHandler(AbstractDataHandler):
    # pylint: disable=abstract-method
    """Abstract File Data Handler."""

    _SCHEMA_MIN_N_PIECES: int = 10
    _REPR_SAMPLE_MIN_N_PIECES: int = 100

    @property
    def reprSampleMinNPieces(self) -> int:   # noqa: N802
        # pylint: disable=invalid-name
        """Minimum number of pieces for reprensetative sample."""
        return self._reprSampleMinNPieces

    @reprSampleMinNPieces.setter
    def reprSampleMinNPieces(self, n: int, /):   # noqa: N802
        # pylint: disable=invalid-name,no-member
        if (n <= self.nPieces) and (n != self._reprSampleMinNPieces):
            self._reprSampleMinNPieces: int = n

    @reprSampleMinNPieces.deleter
    def reprSampleMinNPieces(self):   # noqa: N802
        # pylint: disable=invalid-name,no-member
        self._reprSampleMinNPieces: int = min(self._REPR_SAMPLE_MIN_N_PIECES,
                                              self.nPieces)


class AbstractS3FileDataHandler(AbstractFileDataHandler):
    # pylint: disable=abstract-method
    """Abstract S3 File Data Handler."""

    S3_CLIENT = s3.client()
