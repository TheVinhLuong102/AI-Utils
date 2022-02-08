"""Pandas data processors."""


from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union
from typing import Dict, List, Sequence, Tuple   # Py3.9+: use built-ins

from numpy import array, ndarray
from pandas import DataFrame, Series
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from ..data_types.python import PyPossibleFeatureType
from ..data_types.spark_sql import _STR_TYPE
from ..fs import PathType
from ..iter import to_iterable
from ..namespace import Namespace, DICT_OR_NAMESPACE_TYPES


__all__ = (
    'PandasFlatteningSubsampler',
    'PandasMLPreprocessor',
)


# flake8: noqa
# (too many camelCase names)

# pylint: disable=invalid-name
# e.g., camelCase names


@dataclass(init=True,
           repr=True,
           eq=True,
           order=False,
           unsafe_hash=False,
           frozen=True)
class PandasFlatteningSubsampler:
    """Flattening Subsampler for Pandas Data Frames."""

    columns: Union[str, Tuple[str]]
    everyNRows: int
    totalNRows: int

    @property
    def rowIndexRange(self):
        """Integer row index range."""
        return range(0, self.totalNRows, self.everyNRows)

    @property
    def transformedCols(self) -> List[str]:
        """Flattened column names."""
        r: range = self.rowIndexRange
        return list(chain.from_iterable((f'{col}__{i}' for i in r) for col in self.columns))

    def __call__(self, pandasDF: DataFrame, /) -> Series:
        """Subsample a Pandas Data Frame's certain columns and flatten them."""
        return Series(data=(pandasDF[to_iterable(self.columns, iterable_type=list)]
                            .iloc[self.rowIndexRange].values.flatten(order='F')),
                      index=self.transformedCols,
                      dtype=None, name=None, copy=False, fastpath=False)


class PandasMLPreprocessor:
    # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """ML Preprocessor for Pandas Data Frames."""

    _CAT_INDEX_SCALED_FIELD_NAME: str = '__CAT_IDX_SCALED__'
    _NUM_SCALER_FIELD_NAME: str = '__NUM_SCALER__'

    _PREPROC_CACHE: Dict[Path, Namespace] = {}

    def __init__(self, origToPreprocColMap: Namespace):
        """Init ML Preprocessor."""
        self.origToPreprocColMap: Namespace = origToPreprocColMap

        self.catOrigToPreprocColMap: Namespace = Namespace(**{
            catCol: catPreprocDetails
            for catCol, catPreprocDetails in origToPreprocColMap.items()
            if isinstance(catPreprocDetails, DICT_OR_NAMESPACE_TYPES)
            and (catPreprocDetails['logical-type'] == 'cat')})

        self.sortedCatCols: List[str] = sorted(self.catOrigToPreprocColMap)

        self.sortedCatPreprocCols: List[str] = \
            [self.catOrigToPreprocColMap[catCol]['transform-to']
             for catCol in self.sortedCatCols]

        if self.sortedCatCols:
            self.catIdxScaled: bool = origToPreprocColMap[self._CAT_INDEX_SCALED_FIELD_NAME]

        self.numOrigToPreprocColMap: Namespace = Namespace(**{
            numCol: numPreprocDetails
            for numCol, numPreprocDetails in origToPreprocColMap.items()
            if isinstance(numPreprocDetails, DICT_OR_NAMESPACE_TYPES)
            and (numPreprocDetails['logical-type'] == 'num')})

        self.sortedNumCols: List[str] = sorted(self.numOrigToPreprocColMap)

        self.sortedNumPreprocCols: List[str] = \
            [self.numOrigToPreprocColMap[numCol]['transform-to']
             for numCol in self.sortedNumCols]

        if self.sortedNumCols:
            self.numScaler: Optional[str] = origToPreprocColMap[self._NUM_SCALER_FIELD_NAME]

            if self.numScaler == 'standard':
                self.numScaler: StandardScaler = StandardScaler(copy=True,
                                                                with_mean=True,
                                                                with_std=True)

                # mean value for each feature in the training set
                self.numScaler.mean_ = array([self.numOrigToPreprocColMap[numCol]['mean']
                                              for numCol in self.sortedNumCols])

                # per-feature relative scaling of the data
                self.numScaler.scale_ = array([self.numOrigToPreprocColMap[numCol]['std']
                                               for numCol in self.sortedNumCols])

            elif self.numScaler == 'maxabs':
                self.numScaler: MaxAbsScaler = MaxAbsScaler(copy=True)

                # per-feature maximum absolute value /
                # per-feature relative scaling of the data
                self.numScaler.max_abs_ = self.numScaler.scale_ = \
                    array([self.numOrigToPreprocColMap[numCol]['max-abs']
                           for numCol in self.sortedNumCols])

            elif self.numScaler == 'minmax':
                self.numScaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1),
                                                            copy=True,
                                                            clip=False)

                # per-feature minimum seen in the data
                self.numScaler.data_min_ = array([self.numOrigToPreprocColMap[numCol]['orig-min']
                                                  for numCol in self.sortedNumCols])

                # per-feature maximum seen in the data
                self.numScaler.data_max_ = array([self.numOrigToPreprocColMap[numCol]['orig-max']
                                                  for numCol in self.sortedNumCols])

                # per-feature range (data_max_ - data_min_) seen in the data
                self.numScaler.data_range_ = self.numScaler.data_max_ - self.numScaler.data_min_

                # per-feature relative scaling of the data
                self.numScaler.scale_ = 2 / self.numScaler.data_range_

                # per-feature adjustment for minimum
                self.numScaler.min_ = -1 - (self.numScaler.scale_ * self.numScaler.data_min_)

            else:
                assert self.numScaler is None

            if self.numScaler is not None:
                self.numScaler.feature_names_in_ = self.sortedNumPreprocCols
                self.numScaler.n_features_in_ = len(self.sortedNumCols)

        self.sortedPreprocCols: List[str] = (self.sortedCatPreprocCols +
                                             self.sortedNumPreprocCols)

    def __call__(self, pandasDF: DataFrame, /, *, returnNumPy: bool = False) \
            -> Union[DataFrame, ndarray]:
        # pylint: disable=too-many-locals
        """Preprocess a Pandas Data Frame."""
        _FLOAT_ABS_TOL: float = 1e-9

        if self.sortedCatCols:   # preprocess categorical columns
            for catCol, catPreprocDetails in self.catOrigToPreprocColMap.items():
                nCats: int = catPreprocDetails['n-cats']
                sortedCats: Sequence[PyPossibleFeatureType] = catPreprocDetails['sorted-cats']

                s: Series = pandasDF[catCol]

                pandasDF.loc[:, catPreprocDetails['transform-to']] = (

                    (sum(((s == cat) * i) for i, cat in enumerate(sortedCats)) +
                     ((~s.isin(sortedCats)) * nCats))

                    if catPreprocDetails['physical-type'] == _STR_TYPE

                    else (sum(((s - cat).abs().between(left=0, right=_FLOAT_ABS_TOL) * i)
                              for i, cat in enumerate(sortedCats)) +
                          ((1 - sum(((s - cat).abs().between(left=0, right=_FLOAT_ABS_TOL) * 1)
                                    for cat in sortedCats)) * nCats)))

                # *** NOTE: NumPy BUG ***
                # abs(...) of a data type most negative value equals to the same most negative value
                # github.com/numpy/numpy/issues/5657
                # github.com/numpy/numpy/issues/9463

            if self.catIdxScaled:
                pandasDF.loc[:, self.sortedCatPreprocCols] = minMaxScaledIndices = \
                    2 * pandasDF[self.sortedCatPreprocCols] / nCats - 1

                assert ((minMaxScaledIndices >= -1) & (minMaxScaledIndices <= 1)).all(axis=None), \
                    ValueError('CERTAIN MIN-MAX SCALED INT INDICES '
                               'NOT BETWEEN -1 AND 1: '
                               f'({minMaxScaledIndices.min().min()}, '
                               f'{minMaxScaledIndices.max().max()}) ***')

        if self.sortedNumCols:   # NULL-fill numerical columns
            for numCol, numPreprocDetails in self.numOrigToPreprocColMap.items():
                lowerNull, upperNull = numPreprocDetails['nulls']

                series: Series = pandasDF[numCol]

                checks: Series = series.notnull()

                if lowerNull is not None:
                    checks &= (series > lowerNull)

                if upperNull is not None:
                    checks &= (series < upperNull)

                pandasDF.loc[:, numPreprocDetails['transform-to']] = \
                    series.where(cond=checks,
                                 other=(getattr(series.loc[checks], nullFillMethod)
                                        (axis='index', skipna=True, level=None)
                                        if (nullFillMethod := numPreprocDetails['null-fill-method'])
                                        else numPreprocDetails['null-fill-value']),
                                 inplace=False,
                                 axis=None,
                                 level=None,
                                 errors='raise')

            if self.numScaler:
                pandasDF.loc[:, self.sortedNumPreprocCols] = \
                    self.numScaler.transform(X=pandasDF[self.sortedNumPreprocCols])

        return pandasDF[self.sortedPreprocCols].values if returnNumPy else pandasDF

    @classmethod
    def from_json(cls, path: PathType) -> PandasMLPreprocessor:
        """Load from JSON file."""
        path: Path = Path(path).resolve(strict=True)

        if path not in cls._PREPROC_CACHE:
            cls._PREPROC_CACHE[path] = cls(origToPreprocColMap=Namespace.from_json(path=path))

        return cls._PREPROC_CACHE[path]

    def to_json(self, path: PathType):
        """Save to JSON file."""
        path: Path = Path(path).resolve(strict=True)

        self.origToPreprocColMap.to_json(path=path)

        self._PREPROC_CACHE[path] = self

    @classmethod
    def from_yaml(cls, path: PathType) -> PandasMLPreprocessor:
        """Load from YAML file."""
        path: Path = Path(path).resolve(strict=True)

        if path not in cls._PREPROC_CACHE:
            cls._PREPROC_CACHE[path] = cls(origToPreprocColMap=Namespace.from_yaml(path=path))

        return cls._PREPROC_CACHE[path]

    def to_yaml(self, path: PathType):
        """Save to YAML file."""
        path: Path = Path(path).resolve(strict=True)

        self.origToPreprocColMap.to_yaml(path=path)

        self._PREPROC_CACHE[path] = self
