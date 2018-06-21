import arimo.backend

from arimo.util.aws import key_pair

key, secret = key_pair('PanaAP-CC')


SRC_PTH = 's3a://{}:{}@arimo-panasonic-ap/.arimo/PredMaint/PPP/ErrMults/monthly/DISP_CASE---ex_display_case.parquet'.format(key, secret)
TBL_NAME = 'this_table'
TGT_PTH = '/tmp/tmp.pqt'


if arimo.backend.spark:
    arimo.backend.spark.stop()


arimo.backend.init(
    sparkConf={
        'spark.executor.memoryOverhead': '2g'
        # we can put other configs here to test
    })


src_spark_df = arimo.backend.spark.read.load(
    path=SRC_PTH,
    format='parquet',
    aws_access_key_id=key,
    aws_secret_access_key=secret)


src_spark_df.createOrReplaceTempView(name=TBL_NAME)


cols_to_agg = \
    'inside_temperature', '__RawPred__inside_temperature', \
    'global__MedAE__inside_temperature', 'indiv__MedAE__inside_temperature', \
    'neg__global__MedAE_Mult__inside_temperature', 'pos__global__MedAE_Mult__inside_temperature', \
    'sgn__global__MedAE_Mult__inside_temperature', 'abs__global__MedAE_Mult__inside_temperature', \
    'neg__indiv__MedAE_Mult__inside_temperature', 'pos__indiv__MedAE_Mult__inside_temperature', \
    'sgn__indiv__MedAE_Mult__inside_temperature', 'abs__indiv__MedAE_Mult__inside_temperature', \
    'rowEuclNorm__abs__global__MedAE_Mult', 'rowSumOfLog__abs__global__MedAE_Mult', \
    'rowHigh__abs__global__MedAE_Mult', 'rowLow__abs__global__MedAE_Mult', \
    'rowMean__abs__global__MedAE_Mult', 'rowGMean__abs__global__MedAE_Mult', \
    'rowEuclNorm__abs__indiv__MedAE_Mult', 'rowSumOfLog__abs__indiv__MedAE_Mult', \
    'rowHigh__abs__indiv__MedAE_Mult', 'rowLow__abs__indiv__MedAE_Mult', \
    'rowMean__abs__indiv__MedAE_Mult', 'rowGMean__abs__indiv__MedAE_Mult'


sql_aggs = []

for col_to_agg in cols_to_agg:
    sql_aggs += \
        ['PERCENTILE_APPROX({0}, 0.5) AS dailyMed__{0}'.format(col_to_agg),
         'AVG({0}) AS dailyMean__{0}'.format(col_to_agg),
         'MAX({0}) AS dailyMax__{0}'.format(col_to_agg),
         'MIN({0}) AS dailyMin__{0}'.format(col_to_agg)]


tgt_spark_df = arimo.backend.spark.sql(
    "SELECT \
        equipment_instance_id, \
        date, \
        FIRST(blueprint_uuid) AS blueprint_uuid, \
        {} \
    FROM \
        {} \
    GROUP BY \
        equipment_instance_id, \
        date".format(
    ', '.join(sql_aggs),
    TBL_NAME))


tgt_spark_df.write.save(
    path=TGT_PTH,
    format='parquet',
    partitionBy='date')

# typical errors:

# WARN YarnSchedulerBackend$YarnSchedulerEndpoint:
# Requesting driver to remove executor 2 for reason Container marked as failed:
# container_1524504924019_0006_01_000003 on host: ip-10-26-128-69.ec2.internal.
# Exit status: -100. Diagnostics:Container released on a *lost* node

# ERROR YarnScheduler:
# Lost executor 2 on ip-10-26-128-69.ec2.internal:
# Container marked as failed: container_1524504924019_0006_01_000003 on host: ip-10-26-128-69.ec2.internal.
# Exit status: -100. Diagnostics: Container released on a *lost* node

# WARN TaskSetManager:
# Lost task 8.0 in stage 3.0 (TID 4512, ip-10-26-128-69.ec2.internal, executor 2):
# ExecutorLostFailure (executor 2 exited caused by one of the running tasks)
# Reason: Container marked as failed: container_1524504924019_0006_01_000003 on host: ip-10-26-128-69.ec2.internal.
# Exit status: -100. Diagnostics: Container released on a *lost* node
