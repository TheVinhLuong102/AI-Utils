from __future__ import print_function

import pandas
import pyspark.sql

import arimo.backend


arimo.backend.init()


sdf_0 = arimo.backend.spark.createDataFrame(
    data=pandas.DataFrame(
        data=dict(
            x=[-1, 0, 1])))

sdf_0.cache()
sdf_0.count()


sdf_1 = sdf_0.withColumn('y', pyspark.sql.functions.lit(2))

sdf_1.cache()
sdf_1.count()


# *** UNPERSISTING sdf_0 WILL ALSO UNPERSIST sdf_1 !!! ***
sdf_0.unpersist()
