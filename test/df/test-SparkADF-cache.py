import pandas

import arimo.backend
from arimo.df.spark import SparkADF


adf_0 = SparkADF.create(
    data=pandas.DataFrame(
        data=dict(
            x=[0, 1, 2],
            y=[3, 4, 5],
            z=[6, 7, 8])),
    alias='tbl_0')

adf_0.cache()

assert arimo.backend.spark.catalog.isCached(tableName='tbl_0')


adf_1 = SparkADF.create(
    data=pandas.DataFrame(
        data=dict(
            x=[0, 10, 20],
            y=[30, 40, 50],
            z=[60, 70, 80])),
    alias='tbl_1')

assert not arimo.backend.spark.catalog.isCached(tableName='tbl_1')


adf_0_ = SparkADF.create(
    data=pandas.DataFrame(
        data=dict(
            x=[0, 100, 200],
            y=[300, 400, 500],
            z=[600, 700, 800])),
    alias='tbl_0')

assert not arimo.backend.spark.catalog.isCached(tableName='tbl_0')
