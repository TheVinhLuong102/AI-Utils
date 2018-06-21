import pandas

import pyspark
from pyspark.ml.feature import SQLTransformer


spark = pyspark.sql.SparkSession.builder.getOrCreate()


sdf = spark.createDataFrame(pandas.DataFrame(dict(x=[-1, 0, 1])))

# after below step, SparkUI Storage shows 1 cached RDD
sdf.cache(); sdf.count()

# after below step, cached RDD disappears from SparkUI Storage
new_sdf = SQLTransformer(statement='SELECT * FROM __THIS__').transform(sdf)
