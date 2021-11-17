import logging
import os
import psutil
import subprocess
import sys

import ray
from ray import ray_constants
import pyarrow

import tensorflow
import keras

import pyspark

from h1st_util.util import __path__ as h1st_util_paths
from h1st_util.util.decor import _docstr_verbose
from h1st_util.util.fs import hdfs_client as hdfs, exist, put, rm, \
    _HADOOP_HOME, _HADOOP_CONF_DIR_ENV_VAR_NAME, \
    _ON_LINUX_CLUSTER, _ON_LINUX_CLUSTER_WITH_HDFS
from h1st_util.util.log import STDOUT_HANDLER

from ..aws.ec2.instance_types import INSTANCE_TYPES_INFO, MEMORY_GiB_KEY, N_CPUS_KEY
from .yarn.alloc import optim_alloc


# verify Python version
_MIN_PY_VER = 3, 7

assert sys.version_info >= _MIN_PY_VER, \
    f'*** Python >= {_MIN_PY_VER[0]}.{_MIN_PY_VER[1]} required ***'


_MIN_ARROW_VER = '1.0.1'
assert pyarrow.__version__ >= _MIN_ARROW_VER, \
    f'*** PyArrow >= {_MIN_ARROW_VER} required, but {pyarrow.__version__} installed ***'


_MIN_SPARK_VER = '3.0.1'


_MIN_TF_VER = '2.2.1'   # works with multiprocessing training
assert tensorflow.__version__ >= _MIN_TF_VER, \
    f'*** TensorFlow >= {_MIN_TF_VER} required, but {tensorflow.__version__} installed ***'


_MIN_KERAS_VER = '2.3.1'   # works with multiprocessing training
assert keras.__version__ >= _MIN_KERAS_VER, \
    f'*** Keras >= {_MIN_KERAS_VER} required, but Currently {keras.__version__} installed ***'


# Java Home
_JAVA_HOME_ENV_VAR_NAME = 'JAVA_HOME'

if _ON_LINUX_CLUSTER:
    _JAVA_HOME_ON_H1ST_LINUX_CLUSTER = '/usr/lib/jvm/java-9-openjdk-amd64'

    if not os.path.isdir(_JAVA_HOME_ON_H1ST_LINUX_CLUSTER):
        _JAVA_HOME_ON_H1ST_LINUX_CLUSTER = '/usr/lib/jvm/java-8-openjdk-amd64'
        # assert os.path.isdir(_JAVA_HOME_ON_H1ST_LINUX_CLUSTER)   # TODO

_JAVA_HOME = \
    os.environ.get(
        'JAVA_HOME',
        _JAVA_HOME_ON_H1ST_LINUX_CLUSTER
            if _ON_LINUX_CLUSTER
            else os.popen('echo $(/usr/libexec/java_home)').read()[:-1])


# optimized default Spark configs
_SPARK_HOME_ENV_VAR_NAME = 'SPARK_HOME'

_SPARK_HOME_ON_H1ST_LINUX_CLUSTER = '/opt/spark'
_SPARK_JARS_DIR_PATH_ON_H1ST_LINUX_CLUSTER = os.path.join(_SPARK_HOME_ON_H1ST_LINUX_CLUSTER, 'jars')

_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY = 'spark.files.maxPartitionBytes'
_SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY = 'spark.sql.files.maxPartitionBytes'
_SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY = 'spark.files.openCostInBytes'
_SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY = 'spark.sql.files.openCostInBytes'

_MAX_JAVA_INTEGER = 2 ** 31 - 1
_MAX_JAVA_INTEGER_STR = str(_MAX_JAVA_INTEGER)

_SPARK_CONF = \
    {'fs.s3a.fast.upload': 'true',   # use S3AFastOutputStream to upload data directly from memory

     'spark.default.parallelism': 168,   # set low to reduce scheduling overhead
        # default = 200, but can set > 2000 to activate greater compression and deal better with big data
        # https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/scheduler/MapStatus.scala
        # https://stackoverflow.com/questions/32349611/what-should-be-the-optimal-value-for-spark-sql-shuffle-partitions-or-how-do-we-i

     # 'spark.driver.cores': 1,   # MUST BE SET IN spark-defaults.conf
     # 'spark.driver.maxResultSize': '9g',   # MUST BE SET IN spark-defaults.conf
     # 'spark.driver.memory': '9g',   # MUST BE SET IN spark-defaults.conf

     'spark.dynamicAllocation.enabled': False,   # *** TEMPORARILY DISABLED TO REDUCE DEAD EXECUTORS RISK ***
     'spark.dynamicAllocation.executorIdleTimeout': '600s',
     # 'spark.dynamicAllocation.cachedExecutorIdleTimeout': 'infinity',
        #  MUST set this: executors are idle when streaming cached data
     'spark.dynamicAllocation.initialExecutors': 9,
     # 'spark.dynamicAllocation.maxExecutors': 'infinity',
     'spark.dynamicAllocation.minExecutors': 3,

     # 'spark.executor.cores': ...,   # *** OPTIMALLY SET IN initSpark(...) ***

     'spark.executor.instances': 10 ** 3,   # *** TEMPORARILY DISABLED DynAlloc TO REDUCE DEAD EXECUTORS RISK ***

     # 'spark.executor.memory': '...g',   # *** OPTIMALLY SET IN initSpark(...) ***

     # 'spark.executor.heartbeatInterval': '60s',

     _SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY: 134217728,   # 128 MiB
     _SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY: 134217728,   # 128 MiB
        # The maximum number of bytes to pack into a single partition when reading files.

     _SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY: 4194304,   # 4 MiB
     _SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY: 4194304,   # 4 MiB
        # The estimated cost to open a file, measured by the number of bytes could be scanned in the same time.
        # This is used when putting multiple files into a partition.
        # It is better to over estimate, then the partitions with small files will be faster than partitions with bigger files.
        # https://github.com/apache/spark/blob/master/sql/core/src/main/scala/org/apache/spark/sql/execution/DataSourceScanExec.scala#L412-L453

     'spark.kryo.unsafe': True,
        # Whether to use unsafe based Kryo serializer. Can be substantially faster by using Unsafe Based IO.
     
     'spark.kryoserializer.buffer.max': '2047m',
        # This must be larger than any object you attempt to serialize and must be less than 2048m.

     # WARN SparkConf:66
     # In Spark 1.0 and later spark.local.dir will be overridden by the value set by the cluster manager
     # (via SPARK_LOCAL_DIRS in mesos/standalone and LOCAL_DIRS in YARN)
     # 'spark.local.dir': '/tmp',

     'spark.locality.wait': '3s',   # *** SET LOW ~0 TO INCREASE PARALLELISM, SET HIGH >> 3s TO INCREASE LOCALITY ***

     'spark.memory.fraction': .6,
        # Fraction of (heap space - 300MB) used for execution and storage.
        # The lower this is, the more frequently spills and cached data eviction occur.
        # The purpose of this config is to set aside memory for internal metadata, user data structures,
        # and imprecise size estimation in the case of sparse, unusually large records.
        # Leaving this at the default value is recommended. For more detail, including
        # important information about correctly tuning JVM garbage collection when increasing this value,
        # see https://spark.apache.org/docs/latest/tuning.html#memory-management-overview.

        # expresses the size of M as a fraction of the (JVM heap space - 300MB) (default 0.6).
        # The rest of the space (40%) is reserved for user data structures, internal metadata in Spark,
        # and safeguarding against OOM errors in the case of sparse and unusually large records.
        # The value of spark.memory.fraction should be set in order to fit this amount of heap space comfortably
        # within the JVM's old or 'tenured' generation. See the discussion of advanced GC tuning below for details.

    'spark.memory.storageFraction': .5,
        # Amount of storage memory immune to eviction,
        # expressed as a fraction of the size of the region set aside by spark.memory.fraction.
        # The higher this is, the less working memory may be available to execution and tasks may spill to disk more often.
        # Leaving this at the default value is recommended.
        # For more detail, see https://spark.apache.org/docs/latest/tuning.html#memory-management-overview.

        # expresses the size of R as a fraction of M (default 0.5).
        # R is the storage space within M where cached blocks immune to being evicted by execution.

    # 'spark.memory.offHeap.enabled': True,
        # If true, Spark will attempt to use off-heap memory for certain operations.
        # If off-heap memory use is enabled, then spark.memory.offHeap.size must be positive.

    # 'spark.memory.offHeap.size': '9g',
        # The absolute amount of memory in bytes which can be used for off-heap allocation.
        # This setting has no impact on heap memory usage,
        # so if your executors' total memory consumption must fit within some hard limit then be sure to shrink your JVM heap size accordingly.
        # This must be set to a positive value when spark.memory.offHeap.enabled=true.

     'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',

     'spark.shuffle.minNumPartitionsToHighlyCompress': 2000,
     'spark.shuffle.useOldFetchProtocol': True,   # TODO: fix in Spark 3

     'spark.speculation': True,

     'spark.sql.autoBroadcastJoinThreshold': 68 * (2 ** 20),   # to disable: set to -1

     'spark.sql.catalogImplementation': 'in-memory',

     'spark.sql.codegen.wholeStage': False,   # *** complex Spark SQL code is likely to exceed JVM's 64 KB limit even for 1 stage ***

     'spark.sql.crossJoin.enabled': True,

     'spark.sql.execution.arrow.maxRecordsPerBatch': 10 ** 4,
     'spark.sql.execution.arrow.pyspark.enabled': True,
     'spark.sql.execution.arrow.pyspark.fallback.enabled': True,
     'spark.sql.execution.pandas.convertToArrowArraySafely': True,
        # Arrow safe type check can be disabled by using SQL config `spark.sql.execution.pandas.convertToArrowArraySafely`

     'spark.sql.hive.convertMetastoreParquet': True,
     'spark.sql.hive.filesourcePartitionFileCacheSize': 6 * 10 ** 9,   # default 262144000 bytes = 262 MB

     'spark.sql.inMemoryColumnarStorage.compressed': True,
     'spark.sql.inMemoryColumnarStorage.batchSize': 30000,

     'spark.sql.legacy.timeParserPolicy': 'LEGACY',
     
     # https://community.hortonworks.com/articles/148917/orc-improvements-for-apache-spark-22.html
     'spark.sql.orc.enabled': True,
     'spark.sql.hive.convertMetastoreOrc': True,
     'spark.sql.orc.filterPushdown': True,
     'spark.sql.orc.char.enabled': True,

     'spark.sql.parquet.cacheMetadata': True,   # can speed up querying of static data.
     'spark.sql.parquet.compression.codec': 'snappy',   # others: 'none', 'uncompressed', 'gzip' & 'lzo'
     'spark.sql.parquet.filterPushdown': True,
     'spark.sql.parquet.mergeSchema': True,   # *** NOTE: schema merging is a relatively expensive operation ***

     'spark.sql.shuffle.partitions': 2001,
        # default = 200, but can set > 2000 to activate greater compression and deal better with big data
        # https://github.com/apache/spark/blob/master/core/src/main/scala/org/apache/spark/scheduler/MapStatus.scala
        # https://stackoverflow.com/questions/32349611/what-should-be-the-optimal-value-for-spark-sql-shuffle-partitions-or-how-do-we-i

     'spark.sql.warehouse.dir': '${system:user.dir}/spark-warehouse',

     'spark.sql.sources.default': 'parquet',

     'spark.shuffle.service.enabled': True,

     'spark.ui.enabled': True,
     
     'spark.ui.retainedDeadExecutors': 10,
     'spark.ui.retainedJobs': 100,
     'spark.ui.retainedStages': 100,
     'spark.ui.retainedTasks': 100,

     'spark.sql.ui.retainedExecutions': 100,

     'spark.worker.ui.retainedDrivers': 100,
     'spark.worker.ui.retainedExecutors': 100,

     'spark.yarn.am.memory': '512m',
     'spark.yarn.am.cores': 1}

_SPARK_REPOS = \
    {'http://redshift-maven-repository.s3-website-us-east-1.amazonaws.com/release'}

_SPARK_PKGS = {
    # AWS-related dependencies
    'org.apache.hadoop:hadoop-aws:2.7.2',   # consistent with Hadoop 2.7.2
    'com.amazonaws:aws-java-sdk:1.7.4',   # https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-aws/2.7.2

    # Spark DL Pipelines & TensorFrames
    # 'databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11',
    # 'databricks:tensorframes:0.8.1-s_2.11',

    # JPMML-SparkML
    # 'org.jpmml:jpmml-sparkml:1.3.3',
     
    # MLeap
    # 'ml.combust.mleap:mleap-spark_2.11:0.9.0'
}

_DATA_IO_SPARK_PKGS = \
    dict(avro=
            {# https://spark-packages.org/package/databricks/spark-avro
             'com.databricks:spark-avro_2.11:4.0.0'},

         bigquery=
            {# https://spark-packages.org/package/spotify/spark-bigquery
             'com.spotify:spark-bigquery_2.11:0.1.2'},

         cassandra=
            {# https://spark-packages.org/package/datastax/spark-cassandra-connector
             'datastax:spark-cassandra-connector:1.6.6-s_2.10',
             
             # https://spark-packages.org/package/TargetHolding/pyspark-cassandra
             'TargetHolding:pyspark-cassandra:0.3.5'},

         cloudant=
            {# https://spark-packages.org/package/cloudant-labs/spark-cloudant
             'cloudant-labs:spark-cloudant:2.0.0-s_2.11'},
        
         couch=
            {# https://spark-packages.org/package/couchbaselabs/couchbase-spark-connector   *** DEPRECATED ***
             # 'com.couchbase.client:spark-connector_2.10:1.0.0'

             # https://spark-packages.org/package/couchbase/couchbase-spark-connector
             'com.couchbase.client:spark-connector_2.11:2.1.0'},

         csv=
            {# https://spark-packages.org/package/databricks/spark-csv
             'com.databricks:spark-csv_2.11:1.5.0'},

         delta=
            {'io.delta:delta-core_2.12:0.5.0'},

         elastic=
            {# https://mvnrepository.com/artifact/org.elasticsearch/elasticsearch-hadoop
             # 'org.elasticsearch:elasticsearch-hadoop:6.0.0-alpha-1',   *** ERROR ***

             # https://spark-packages.org/package/elastic/elasticsearch-hadoop
             'org.elasticsearch:elasticsearch-spark-20_2.11:5.3.1',

             # https://spark-packages.org/package/TargetHolding/pyspark-elastic
             'TargetHolding:pyspark-elastic:0.4.2'},

         geospatial=
            {# https://spark-packages.org/package/harsha2010/magellan
             'harsha2010:magellan:1.0.4-s_2.11'},

         githubpr=
            {# https://spark-packages.org/package/lightcopy/spark-github-pr
             'lightcopy:spark-github-pr:1.3.0-s_2.10'},

         gsheet=
            {# https://spark-packages.org/package/potix2/spark-google-spreadsheets
             'com.github.potix2:spark-google-spreadsheets_2.11:0.5.0'},

         # *** java.io.FileNotFoundException: File file:/home/h1st/.ivy2/jars/org.apache.zookeeper_zookeeper-3.4.6.jar does not exist ***
         # hadoopcryptoledger=
         #    {# https://spark-packages.org/package/ZuInnoTe/spark-hadoopcryptoledger-ds
         #     'com.github.zuinnote:spark-hadoopcryptoledger-ds_2.11:1.2.0'},

         # *** java.io.FileNotFoundException: File file:/home/h1st/.ivy2/jars/org.apache.zookeeper_zookeeper-3.4.6.jar does not exist ***
         # hadoopoffice=
         #    {# https://spark-packages.org/package/ZuInnoTe/spark-hadoopoffice-ds
         #     'com.github.zuinnote:spark-hadoopoffice-ds_2.11:1.1.1'},

         hazelcast=
            {# https://spark-packages.org/package/erenavsarogullari/spark-hazelcast-connector
             'erenavsarogullari:spark-hazelcast-connector:1.0.0-s_2.11'},

         hbase=
            {# https://github.com/hortonworks-spark/shc
             # 'com.hortonworks:shc-core:1.1.1-2.1-s_2.11',

             # https://spark-packages.org/package/nerdammer/spark-hbase-connector
             'it.nerdammer.bigdata:spark-hbase-connector_2.10:1.0.3',

             # https://spark-packages.org/package/zhzhan/shc
             'zhzhan:shc:0.0.11-1.6.1-s_2.10'},

         infinispan=
            {# https://spark-packages.org/package/infinispan/infinispan-spark
             'org.infinispan:infinispan-spark_2.11:0.5'},

         iqmulus=
            {# https://spark-packages.org/package/IGNF/spark-iqmulus
             'IGNF:spark-iqmulus:0.1.0-s_2.10'},

         mongo=
            {# https://spark-packages.org/package/mongodb/mongo-spark
             'org.mongodb.spark:mongo-spark-connector_2.10:2.0.0',

             # https://spark-packages.org/package/Stratio/spark-mongodb
             'com.stratio.datasource:spark-mongodb_2.11:0.12.0'},

         neo=
            {# https://spark-packages.org/package/neo4j-contrib/neo4j-spark-connector
             'neo4j-contrib:neo4j-spark-connector:2.0.0-M2'},

         netflow=
            {# https://spark-packages.org/package/sadikovi/spark-netflow
             'com.github.sadikovi:spark-netflow_2.10:2.0.1'},

         pg={# https://mvnrepository.com/artifact/org.postgresql/postgresql
             'org.postgresql:postgresql:42.2.5'},

         powerbi=
            {# https://spark-packages.org/package/granturing/spark-power-bi
             'com.granturing:spark-power-bi_2.10:1.5.0_0.0.7'},

         protobuf=
            {# https://spark-packages.org/package/saurfang/sparksql-protobuf
             'saurfang:sparksql-protobuf:0.1.2-s_2.10'},

         redshift=
            {# https://mvnrepository.com/artifact/com.amazonaws/aws-java-sdk-redshift
             'com.amazonaws:aws-java-sdk-redshift:1.11.231',

             # http://docs.aws.amazon.com/redshift/latest/mgmt/configure-jdbc-connection-with-maven.html
             'com.amazon.redshift:redshift-jdbc42:1.2.10.1009',   # *** !!! NOT FOUND !!! ***
                
             # https://spark-packages.org/package/databricks/spark-redshift
             'com.databricks:spark-redshift_2.11:3.0.0-preview1'},

         riak=
            {# https://spark-packages.org/package/basho/spark-riak-connector
             'com.basho.riak:spark-riak-connector_2.10:1.6.3'},

         ryft=
            {# https://spark-packages.org/package/getryft/spark-ryft-connector
             'com.ryft:spark-ryft-connector_2.10:0.9.0'},

         salesforce=
            {# https://spark-packages.org/package/springml/spark-salesforce
             'com.springml:spark-salesforce_2.11:1.1.0'},

         sas=
            {# https://spark-packages.org/package/saurfang/spark-sas7bdat
             'saurfang:spark-sas7bdat:1.1.5-s_2.11'},

         sequoia=
            {# https://spark-packages.org/package/SequoiaDB/spark-sequoiadb
             'SequoiaDB:spark-sequoiadb:1.12-s_2.11'},

         sftp=
            {# https://spark-packages.org/package/springml/spark-sftp
             'com.springml:spark-sftp_2.11:1.1.1'},

         snappydata=
            {# https://spark-packages.org/package/SnappyDataInc/snappydata
             'SnappyDataInc:snappydata:0.8-s_2.11'},

         # solr=
         #    {# https://spark-packages.org/package/LucidWorks/spark-solr
         #     'com.lucidworks.spark:spark-solr:2.0.1'},

         sparql=
            {# https://spark-packages.org/package/USU-Research/spark-sparql-connector
             'USU-Research:spark-sparql-connector:1.0.0-beta1-s_2.10'},

         stocator=
            {# https://spark-packages.org/package/SparkTC/stocator
             'com.ibm.stocator:stocator:1.0.8'},

         succint=
            {# https://spark-packages.org/package/amplab/succinct
             'amplab:succinct:0.1.7'},

         tf={},

         xml=
            {# https://spark-packages.org/package/HyukjinKwon/spark-xml
             'HyukjinKwon:spark-xml:0.1.1-s_2.10'},

         # *** java.io.FileNotFoundException: File file:/home/h1st/.ivy2/jars/net.java.dev.jna_jna-3.3.0.jar does not exist ***
         # zzz=
         #    {# https://spark-packages.org/package/Stratio/spark-crossdata
         #     'Stratio:spark-crossdata:1.4.0',

             # https://spark-packages.org/package/Stratio/Datasource-Receiver
         #     'com.stratio.receiver:spark-datasource_1.6:0.1.0',

             # https://spark-packages.org/package/alvsanand/spark-generic-connector
         #     'alvsanand:spark-generic-connector:0.2.0-spark_2x-s_2.11',

             # https://spark-packages.org/package/cookieai/cookie-datasets
         #     'ai.cookie:cookie-datasets_2.10:0.1.0'}
        )

DATA_IO_OPTIONS = set(_DATA_IO_SPARK_PKGS)


# YARN
_YARN_CONF_DIR_ENV_VAR_NAME = 'YARN_CONF_DIR'

_YARN_JARS_DIR_NAME = '/YARN-JARs'
_YARN_JARS_DIR_PATH = 'hdfs://{}'.format(_YARN_JARS_DIR_NAME)


# un-initialized SparkSession
spark = None

_BASE_DIR_PATH = os.path.dirname(__file__)

_SPARK_DEPS_DIR_PATH = \
    os.path.join(
        _BASE_DIR_PATH,
        '_SparkDeps')

_SPARK_JARS_DIR_PATH = \
    os.path.join(
        _SPARK_DEPS_DIR_PATH,
        'JARs')

_SPARK_PY_FILES_DIR_PATH = \
    os.path.join(
        _SPARK_DEPS_DIR_PATH,
        'PyFiles')

_SPARK_H1ST_PACKAGE_PY_FILE_PATHS = os.path.join(h1st_util_paths[0], 'dl.py'),

_SPARK_CKPT_DIR = '/tmp/.spark/ckpt'


def chkRay() -> bool:
    return ray.is_initialized()


def initRay(*, verbose: bool = False) -> None:
    ray.init(
        address=None,
            # str
            # The address of the Ray cluster to connect to.
            # If this address is not provided, then this command will start Redis,
            # a raylet, a plasma store, a plasma manager, and some workers.
            # It will also kill these processes when Python exits.
            # If the driver is running on a node in a Ray cluster,
            # using auto as the value tells the driver to detect the the cluster,
            # removing the need to specify a specific node address.

        redis_port=None,
            # int
            # The port that the primary Redis shard should listen to.
            # If None, then a random port will be chosen.

        num_cpus=psutil.cpu_count(logical=True),
            # int
            # Number of CPUs the user wishes to assign to each raylet.

        num_gpus=0,
            # int
            # Number of GPUs the user wishes to assign to each raylet.

        memory=None,
            # The amount of memory (in bytes) that is available for use by workers requesting memory resources.
            # By default, this is automatically set based on available system memory.
        
        object_store_memory=None,
            # The amount of memory (in bytes) to start the object store with.
            # By default, this is automatically set based on available system memory, subject to a 20GB cap.

        resources=None,
            # A dictionary mapping the names of custom resources to the quantities for them available.

        driver_object_store_memory=None,
            # int
            # Limit the amount of memory the driver can use in the object store for creating objects.
            # By default, this is autoset based on available system memory, subject to a 20GB cap.

        redis_max_memory=None,
            # The max amount of memory (in bytes) to allow each redis shard to use.
            # Once the limit is exceeded, redis will start LRU eviction of entries.
            # This only applies to the sharded redis tables (task, object, and profile tables).
            # By default, this is autoset based on available system memory, subject to a 10GB cap.

        log_to_driver=verbose,   # reduce verbosity
            # bool
            # If true, the output from all of the worker processes on all nodes will be directed to the driver.

        node_ip_address=ray_constants.NODE_DEFAULT_IP,
            # str
            # The IP address of the node that we are on.

        object_ref_seed=None,
            # int
            # Used to seed the deterministic generation of object refs.
            # The same value can be used across multiple runs of the same driver
            # in order to generate the object refs in a consistent manner.
            # However, the same ID should not be used for different drivers.

        local_mode=False,
            # bool
            # If true, the code will be executed serially.
            # This is useful for debugging.

        redirect_worker_output=None,

        redirect_output=None,

        ignore_reinit_error=False,
            # If true, Ray suppresses errors from calling ray.init() a second time.
            # Ray won’t be restarted.

        num_redis_shards=None,
            # The number of Redis shards to start in addition to the primary Redis shard.

        redis_max_clients=None,
            # If provided, attempt to configure Redis with this maxclients number.

        # redis_password=ray_constant.REDIS_DEFAULT_PASSWORD,
            # str
            # Prevents external clients without the password from connecting to Redis if provided.

        plasma_directory=None,
            # A directory where the Plasma memory mapped files will be created.

        huge_pages=False,
            # Boolean flag indicating whether to start the Object Store with hugetlbfs support.
            # Requires plasma_directory.

        include_java=False,
            # Boolean flag indicating whether or not to enable java workers.

        include_dashboard=None,
            # Boolean flag indicating whether or not to start the Ray dashboard,
            # which displays the status of the Ray cluster.
            # If this argument is None, then the UI will be started if the relevant dependencies are present.

        dashboard_host='localhost',
            # The host to bind the dashboard server to.
            # Can either be localhost (127.0.0.1) or 0.0.0.0 (available from all interfaces).
            # By default, this is set to localhost to prevent access from external machines.

        dashboard_port=ray_constants.DEFAULT_DASHBOARD_PORT,
            # The port to bind the dashboard server to.
            # Defaults to 8265.

        job_id=None,   # unexpected keyword argument
            # The ID of this job.

        configure_logging=True,
            # True (default) if configuration of logging is allowed here.
            # Otherwise, the user may want to configure it separately.

        logging_level=logging.INFO,
            # Logging level, defaults to logging.INFO.
            # Ignored unless “configure_logging” is true.

        logging_format=ray_constants.LOGGER_FORMAT,
            # Logging format, defaults to string containing a timestamp, filename, line number, and message.
            # See the source file ray_constants.py for details. Ignored unless “configure_logging” is true.

        plasma_store_socket_name=None,
            # If provided, specifies the socket name used by the plasma store.

        raylet_socket_name=None,
            # If provided, specifies the socket path used by the raylet process.

        temp_dir=None,
            # If provided, specifies the root temporary directory for the Ray process.
            # Defaults to an OS-specific conventional location, e.g., “/tmp/ray”.

        load_code_from_local=False,
            # Whether code should be loaded from a local module or from the GCS.

        java_worker_options=None,
            # Overwrite the options to start Java workers.

        _internal_config=None,
            # str
            # JSON configuration for overriding RayConfig defaults.
            # For testing purposes ONLY.

        lru_evict=False,
            # bool
            # If True, when an object store is full, it will evict objects in LRU order
            # to make more space and when under memory pressure, ray.UnreconstructableError may be thrown.
            # If False, then reference counting will be used to decide which objects are safe to evict
            # and when under memory pressure, ray.ObjectStoreFullError may be thrown.

        enable_object_reconstruction=False
            # bool
            # If True, when an object stored in the distributed plasma store is lost due to node failure,
            # Ray will attempt to reconstruct the object by re-executing the task that created the object.
            # Arguments to the task will be recursively reconstructed.
            # If False, then ray.UnreconstructableError will be thrown.
    )


def chkSpark():
    global spark
    return spark and spark._instantiatedSession and spark.sparkContext._active_spark_context and spark.sparkContext._jsc


def updateYARNJARs():
    if _ON_LINUX_CLUSTER_WITH_HDFS:
        put(from_local=_SPARK_JARS_DIR_PATH_ON_H1ST_LINUX_CLUSTER,
            to_hdfs=_YARN_JARS_DIR_NAME,
            is_dir=True,
            _mv=False,
            hadoop_home=_HADOOP_HOME)


def rmSparkCkPts():
    """
    Clean up Spark Checkpoint directory
    """
    rm(path=_SPARK_CKPT_DIR,
       hdfs=_ON_LINUX_CLUSTER_WITH_HDFS,
       is_dir=True,
       hadoop_home=_HADOOP_HOME)


@_docstr_verbose
def initSpark(
        sparkApp=None,
        sparkHome=os.environ.get(_SPARK_HOME_ENV_VAR_NAME, _SPARK_HOME_ON_H1ST_LINUX_CLUSTER),
        sparkConf={},
        sparkRepos=(),
        sparkPkgs=(),
        javaHome=None,
        hadoopConfDir=None,
        yarnConfDir=None,
        yarnUpdateJARs=False,
        dataIO={'avro', 'pg', 'redshift', 'sftp'},
        executor_aws_ec2_instance_type='c5n.9xlarge'):
    """
    Launch new ``SparkSession`` or connect to existing one, and binding it to ``h1st.data_backend.spark``

    Args:
        sparkApp (str): name to give to the ``SparkSession`` to be launched

        sparkHome (str): path to Spark installation, if not already set in ``SPARK_HOME`` environment variable

        sparkConf (tuple/list): tuple/list of configs to over-ride default Spark configs

        sparkPkgs (tuple/list of str): tuple/list of Maven and/or Spark packages with which to launch Spark

        javaHome (str): path to Java Development Kit (JDK), if not already set in ``JAVA_HOME`` environment variable

        hadoopConfDir (str): path to Hadoop configuration directory;
            *ignored* if not running on a YARN cluster or if Hadoop is installed at ``/opt/hadoop``

        ckptDir (str): path to default Spark checkpoint directory

        dataIO (set): additional data IO support options
    """
    assert (pyspark.__version__ >= _MIN_SPARK_VER), \
        f'*** Spark >= {_MIN_SPARK_VER} required, but {pyspark.__version__} installed ***'

    # initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(STDOUT_HANDLER)

    # driver Python executable path
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'python3'
    # worker Python executable path
    os.environ['PYSPARK_PYTHON'] = '/opt/miniconda3/bin/python3'

    # set relevant environment variables for Java, Spark, Hadoop & YARN
    if javaHome:
        os.environ[_JAVA_HOME_ENV_VAR_NAME] = javaHome

    elif _JAVA_HOME:
        os.environ[_JAVA_HOME_ENV_VAR_NAME] = _JAVA_HOME

    if sparkHome:
        os.environ[_SPARK_HOME_ENV_VAR_NAME] = sparkHome

    if _ON_LINUX_CLUSTER_WITH_HDFS:
        os.environ[_HADOOP_CONF_DIR_ENV_VAR_NAME] = \
            hadoopConfDir \
            if hadoopConfDir \
            else os.environ.get(
                _HADOOP_CONF_DIR_ENV_VAR_NAME,
                os.path.join(_HADOOP_HOME, 'conf')
                    if _HADOOP_HOME
                    else None)

        if yarnConfDir:
            os.environ[_YARN_CONF_DIR_ENV_VAR_NAME] = yarnConfDir

    os.environ['PYSPARK_SUBMIT_ARGS'] = \
        '--py-files {} --repositories {} --packages {} pyspark-shell'.format(
            # ','.join(
            #     os.path.join(_SPARK_JARS_DIR_PATH, jar_file_name)
            #     for jar_file_name in os.listdir(_SPARK_JARS_DIR_PATH)
            #     if jar_file_name.endswith('.jar')),

            ','.join(_SPARK_H1ST_PACKAGE_PY_FILE_PATHS),

            ','.join(_SPARK_REPOS.union(sparkRepos)),

            ','.join(
                _SPARK_PKGS.union(
                    sparkPkgs,
                    *(_DATA_IO_SPARK_PKGS[dataIOOption.lower()]
                      for dataIOOption in dataIO))))

    # set / create SparkSession
    global spark

    if spark:
        assert spark._instantiatedSession is None
        assert spark.sparkContext._active_spark_context is None
        assert spark.sparkContext._jsc is None
            
    # build Spark Configs
    conf = \
        pyspark.SparkConf() \
        .setAppName(
            sparkApp
            if sparkApp
            else os.getcwd())

    _sparkConf = _SPARK_CONF.copy()

    _sparkConf.update(sparkConf)

    # optimally allocating YARN containers
    executor_aws_ec2_instance_type_info = \
        INSTANCE_TYPES_INFO.loc[executor_aws_ec2_instance_type]

    optim_alloc_details = \
        optim_alloc(
            node_mem_gib=executor_aws_ec2_instance_type_info[MEMORY_GiB_KEY])

    n_executors_per_node = optim_alloc_details['n_executors']

    _sparkConf['spark.executor.memory'] = \
        mem_gib_per_executor = \
        f"{optim_alloc_details['executor_mem_gib']}g"

    _sparkConf['spark.executor.cores'] = \
        n_cpus_per_executor = \
        int(1.68 *   # over-allocating CPUs to maximize CPU usage
            executor_aws_ec2_instance_type_info[N_CPUS_KEY] / n_executors_per_node)

    logger.info(
        msg='Allocating {:,}x {} {:,}-CPU Executors per {} ({}-GiB {:,}-CPU) YARN Worker Node (Leaving {:.1f} GiB for Driver)...'.format(
                n_executors_per_node, mem_gib_per_executor, n_cpus_per_executor,
                executor_aws_ec2_instance_type,
                executor_aws_ec2_instance_type_info[MEMORY_GiB_KEY], executor_aws_ec2_instance_type_info[N_CPUS_KEY],
                optim_alloc_details['avail_for_driver_mem_gib']))

    if _ON_LINUX_CLUSTER_WITH_HDFS:
        if exist(path=_YARN_JARS_DIR_NAME,
                 hdfs=True,
                 dir=True):
            if not yarnUpdateJARs:
                # *** TODO: FIX ***
                # _sparkConf['spark.yarn.jars'] = _YARN_JARS_DIR_NAME
                pass

    else:
        yarnUpdateJARs = False

    for k, v in _sparkConf.items():
        conf.set(k, v)

    # remove any existing derby.log & metastore_db to avoid Hive start-up errors
    rm(path='derby.log',
       hdfs=False,
       is_dir=False)

    rm(path='metastore_db',
       hdfs=False,
       is_dir=True)

    # clean up existing Spark checkpoints
    rmSparkCkPts()

    # get / create SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .config(conf=conf) \
        .enableHiveSupport() \
        .getOrCreate()

    logger.info(msg='SparkSession = {}'.format(spark))

    # BELOW DOESN'T WORK FOR dev/preview VERSIONS
    # assert spark.version == pyspark.__version__, \
    #     '*** PySpark v{} does not match underlying Spark v{} ***'.format(pyspark.__version__, spark.version)

    spark.sparkContext.setLogLevel('WARN')   # ALL, DEBUG, ERROR, FATAL, INFO, OFF, TRACE or WARN

    spark.sparkContext.setCheckpointDir(dirName=_SPARK_CKPT_DIR)

    # set Hadoop Conf in Spark Context
    if os.environ.get('AWS_ACCESS_KEY_ID'):
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.awsAccessKeyId", os.environ.get('AWS_ACCESS_KEY_ID'))
        spark.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.awsSecretAccessKey", os.environ.get('AWS_SECRET_ACCESS_KEY'))

    # register Uder-Defined Functions (UDFs)
    from pyspark.ml.linalg import DenseVector, VectorUDT
    from pyspark.sql.types import ArrayType, DoubleType

    spark.udf.register(
        name='_ARRAY_TO_VECTOR',
        f=lambda a: DenseVector(a),
        returnType=VectorUDT())

    spark.udf.register(
        name='_VECTOR_TO_ARRAY',
        f=lambda v: v.array.tolist(),
        returnType=ArrayType(DoubleType()))

    if yarnUpdateJARs:
        msg = 'Putting JARs from {} to {}...'.format(
            _SPARK_JARS_DIR_PATH_ON_H1ST_LINUX_CLUSTER,
            _YARN_JARS_DIR_PATH)
        logger.info(msg)
        updateYARNJARs()
        logger.info(msg + ' done!')


def setSpark1Partition1File(on=True):
    assert chkSpark()

    global spark

    spark_files_maxPartitionBytes = \
        spark.conf.get(_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY)

    if on:
        if spark_files_maxPartitionBytes == str(_SPARK_CONF[_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY]):
            spark.conf.set(
                _SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY,
                _MAX_JAVA_INTEGER)

            spark.conf.set(
                _SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY,
                _MAX_JAVA_INTEGER)

            spark.conf.set(
                _SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY,
                _MAX_JAVA_INTEGER)

            spark.conf.set(
                _SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY,
                _MAX_JAVA_INTEGER)

            assert spark.conf.get(_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY) \
                == _MAX_JAVA_INTEGER_STR

        else:
            assert spark_files_maxPartitionBytes \
                == _MAX_JAVA_INTEGER_STR

        assert spark.conf.get(_SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY) \
            == _MAX_JAVA_INTEGER_STR

        assert spark.conf.get(_SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY) \
            == _MAX_JAVA_INTEGER_STR

        assert spark.conf.get(_SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY) \
            == _MAX_JAVA_INTEGER_STR

    else:
        if spark_files_maxPartitionBytes == _MAX_JAVA_INTEGER_STR:
            spark.conf.set(
                _SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY,
                _SPARK_CONF[_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY])

            spark.conf.set(
                _SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY,
                _SPARK_CONF[_SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY])

            spark.conf.set(
                _SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY,
                _SPARK_CONF[_SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY])

            spark.conf.set(
                _SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY,
                _SPARK_CONF[_SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY])

            assert spark.conf.get(_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY) \
                == str(_SPARK_CONF[_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY])

        else:
            assert spark_files_maxPartitionBytes \
                == str(_SPARK_CONF[_SPARK_FILES_MAX_PARTITION_BYTES_CONFIG_KEY])

        assert spark.conf.get(_SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY) \
            == str(_SPARK_CONF[_SPARK_SQL_FILES_MAX_PARTITION_BYTES_CONFIG_KEY])

        assert spark.conf.get(_SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY) \
            == str(_SPARK_CONF[_SPARK_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY])

        assert spark.conf.get(_SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY) \
            == str(_SPARK_CONF[_SPARK_SQL_FILES_OPEN_COST_IN_BYTES_CONFIG_KEY])


def runSparkWorkerCmd(cmd, n=9):
    if _ON_LINUX_CLUSTER_WITH_HDFS:
        def run(_):
            return subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE) \
                .communicate()

        global spark
        if not spark:
            initSpark()

        results = spark.sparkContext \
            .parallelize(range(n)) \
            .map(run) \
            .collect()

        for i, result in enumerate(results):
            print('*** RESULT {} / {} ***'.format(i + 1, n))
            print('*** STDOUT ***', result[0], sep='\n')
            print('*** STDERR ***', result[1], sep='\n')

        return results


def installSparkWorkerDeps(*deps, **kwargs):
    if _ON_LINUX_CLUSTER_WITH_HDFS:
        def installDeps(_, deps=deps):
            import os
            os.system('pip install --upgrade {}'.format(' '.join(deps)))

        global spark
        if not spark:
            initSpark()

        return spark.sparkContext \
                .parallelize(range(kwargs.get('n', 30))) \
                .map(installDeps) \
                .collect()
