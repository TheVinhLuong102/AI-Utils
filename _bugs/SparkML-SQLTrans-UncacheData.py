import pandas

import pyspark
from pyspark.ml.feature import SQLTransformer


spark = pyspark.sql.SparkSession.builder.getOrCreate()


df = spark.createDataFrame(pandas.DataFrame(dict(x=[-1, 0, 1])))

# after below step, SparkUI Storage shows 1 cached RDD
df.cache(); df.count()

# after below step, cached RDD disappears from SparkUI Storage
new_df = SQLTransformer(statement='SELECT * FROM __THIS__').transform(df)
