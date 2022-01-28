"""Pandas data processors."""


from numpy import array, hstack
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler

from ..data_types.spark_sql import _STR_TYPE


__all__ = 'PandasNullFiller', 'PandasMLPreprocessor'


class PandasNullFiller:
    def __init__(self, nullFillDetails):
        self.nullFillDetails = nullFillDetails

    def __call__(self, pandasDF):
        for col, nullFillColNameNDetails in self.nullFillDetails.items():
            if (col not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and \
                    isinstance(nullFillColNameNDetails, list) and \
                    (len(nullFillColNameNDetails) == 2):
                _, nullFill = nullFillColNameNDetails

                lowerNull, upperNull = nullFill['Nulls']

                series = pandasDF[col]

                chks = series.notnull()

                if lowerNull is not None:
                    chks &= (series > lowerNull)

                if upperNull is not None:
                    chks &= (series < upperNull)

                pandasDF.loc[
                    :,
                    (AbstractDataHandler._NULL_FILL_PREFIX +
                     col +
                     AbstractDataHandler._PREP_SUFFIX)] = \
                    series.where(
                        cond=chks,
                        other=nullFill['NullFillValue'],
                        inplace=False,
                        axis=None,
                        level=None,
                        errors='raise',
                        try_cast=False)
                # ^^^ SettingWithCopyWarning (?)
                # A value is trying to be set
                # on a copy of a slice from a DataFrame.
                # Try using .loc[row_indexer,col_indexer] = value instead

        return pandasDF


class PandasMLPreprocessor:
    def __init__(self,
                 addCols,
                 typeStrs,
                 catOrigToPrepColMap,
                 numOrigToPrepColMap,
                 returnNumPyForCols=None):
        self.addCols = addCols

        self.typeStrs = typeStrs

        assert not catOrigToPrepColMap['__OHE__']
        self.catOrigToPrepColMap = catOrigToPrepColMap
        self.scaleCat = catOrigToPrepColMap['__SCALE__']

        self.numNullFillPandasDFTransform = \
            PandasNullFiller(
                nullFillDetails=numOrigToPrepColMap)

        self.numNullFillCols = []
        self.numPrepCols = []
        self.numPrepDetails = []

        for numCol, numPrepColNDetails in numOrigToPrepColMap.items():
            if (numCol not in ('__TS_WINDOW_CLAUSE__', '__SCALER__')) and \
                    isinstance(numPrepColNDetails, list) and \
                    (len(numPrepColNDetails) == 2):
                self.numNullFillCols.append(
                    AbstractDataHandler._NULL_FILL_PREFIX +
                    numCol +
                    AbstractDataHandler._PREP_SUFFIX)

                numPrepCol, numPrepDetails = numPrepColNDetails
                self.numPrepCols.append(numPrepCol)
                self.numPrepDetails.append(numPrepDetails)

        if returnNumPyForCols:
            self.returnNumPyForCols = \
                to_iterable(returnNumPyForCols, iterable_type=list)

            nCatCols = len(catOrigToPrepColMap)
            self.catPrepCols = returnNumPyForCols[:nCatCols]

            numPrepCols = returnNumPyForCols[nCatCols:]
            numPrepColListIndices = \
                [numPrepCols.index(numPrepCol)
                 for numPrepCol in self.numPrepCols]
            self.numNullFillCols = \
                [self.numNullFillCols[i]
                 for i in numPrepColListIndices]
            self.numPrepCols = \
                [self.numPrepCols[i]
                 for i in numPrepColListIndices]
            self.numPrepDetails = \
                [self.numPrepDetails[i]
                 for i in numPrepColListIndices]

        else:
            self.returnNumPyForCols = None

        self.numScaler = numOrigToPrepColMap['__SCALER__']

        if self.numScaler == 'standard':
            self.numScaler = \
                StandardScaler(
                    copy=True,
                    with_mean=True,
                    with_std=True)

            # mean value for each feature in the training set
            self.numScaler.mean_ = \
                array(
                    [numPrepDetails['Mean']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature relative scaling of the data
            self.numScaler.scale_ = \
                array(
                    [numPrepDetails['StdDev']
                     for numPrepDetails in self.numPrepDetails])

        elif self.numScaler == 'maxabs':
            self.numScaler = \
                MaxAbsScaler(
                    copy=True)

            # per-feature maximum absolute value /
            # per-feature relative scaling of the data
            self.numScaler.max_abs_ = \
                self.numScaler.scale_ = \
                array(
                    [numPrepDetails['MaxAbs']
                     for numPrepDetails in self.numPrepDetails])

        elif self.numScaler == 'minmax':
            self.numScaler = \
                MinMaxScaler(
                    feature_range=(-1, 1),
                    copy=True)

            # per-feature minimum seen in the data
            self.numScaler.data_min_ = \
                array(
                    [numPrepDetails['OrigMin']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature maximum seen in the data
            self.numScaler.data_max_ = \
                array(
                    [numPrepDetails['OrigMax']
                     for numPrepDetails in self.numPrepDetails])

            # per-feature range (data_max_ - data_min_) seen in the data
            self.numScaler.data_range_ = \
                self.numScaler.data_max_ - self.numScaler.data_min_

            # per-feature relative scaling of the data
            self.numScaler.scale_ = \
                2 / self.numScaler.data_range_

            # per-feature adjustment for minimum
            self.numScaler.min_ = \
                -1 - (self.numScaler.scale_ * self.numScaler.data_min_)

        else:
            assert self.numScaler is None

        if self.numScaler is not None:
            self.numScaler.n_features_in_ = len(self.numPrepDetails)

    def __call__(self, pandasDF):
        _FLOAT_ABS_TOL = 1e-9

        for col, value in self.addCols.items():
            pandasDF[col] = value

        for catCol, prepCatColNameNDetails in self.catOrigToPrepColMap.items():
            if (catCol not in ('__OHE__', '__SCALE__')) and \
                    isinstance(prepCatColNameNDetails, list) and \
                    (len(prepCatColNameNDetails) == 2):
                prepCatCol, catColDetails = prepCatColNameNDetails

                cats = catColDetails['Cats']
                nCats = catColDetails['NCats']

                s = pandasDF[catCol]

                pandasDF.loc[:, prepCatCol] = \
                    (sum(((s == cat) * i)
                         for i, cat in enumerate(cats)) +
                     ((~s.isin(cats)) * nCats)) \
                    if self.typeStrs[catCol] == _STR_TYPE \
                    else (sum(((s - cat).abs().between(left=0,
                                                       right=_FLOAT_ABS_TOL,
                                                       inclusive='both') * i)
                              for i, cat in enumerate(cats)) +
                          ((1 -
                            sum((s - cat).abs().between(left=0,
                                                        right=_FLOAT_ABS_TOL,
                                                        inclusive='both')
                                for cat in cats)) *
                           nCats))
                # *** NOTE NumPy BUG ***
                # *** abs(...) of a data type most negative value equals to
                # the same most negative value ***
                # https://github.com/numpy/numpy/issues/5657
                # https://github.com/numpy/numpy/issues/9463
                # http://numpy-discussion.10968.n7.nabble.com/abs-for-max-negative-integers-desired-behavior-td8939.html

                # ^^^ SettingWithCopyWarning (?)
                # A value is trying to be set on
                # a copy of a slice from a DataFrame.
                # Try using .loc[row_indexer,col_indexer] = value instead

                if self.scaleCat:
                    pandasDF.loc[:, prepCatCol] = minMaxScaledIdxSeries = \
                        2 * pandasDF[prepCatCol] / nCats - 1
                    # ^^^ SettingWithCopyWarning (?)
                    # A value is trying to be set on
                    # a copy of a slice from a DataFrame.
                    # Try using .loc[row_indexer,col_indexer] = value instead

                    assert minMaxScaledIdxSeries.between(
                        left=-1, right=1, inclusive='both').all(), \
                        (f'*** "{prepCatCol}" ({nCats:,} CATS) '
                         'CERTAIN MIN-MAX SCALED INT INDICES '
                         'NOT BETWEEN -1 AND 1: '
                         f'({minMaxScaledIdxSeries.min()}, '
                         f'{minMaxScaledIdxSeries.max()}) ***')

        pandasDF = \
            self.numNullFillPandasDFTransform(
                pandasDF=pandasDF)

        if self.returnNumPyForCols:
            return (hstack(
                    (pandasDF[self.catPrepCols].values,
                     self.numScaler.transform(
                         X=pandasDF[self.numNullFillCols])))
                    if self.numScaler
                    else pandasDF[self.returnNumPyForCols].values)

        if self.numScaler:
            pandasDF[self.numPrepCols] = \
                DataFrame(
                    data=self.numScaler.transform(
                        X=pandasDF[self.numNullFillCols]))
            # ^^^ SettingWithCopyWarning (?)
            # A value is trying to be set
            # on a copy of a slice from a DataFrame.
            # Try using .loc[row_indexer,col_indexer] = value instead

        return pandasDF
