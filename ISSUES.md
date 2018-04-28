# Technical Issues to Note

## `Spark`

### 0. Unresolved Bugs/Problems

- Data & tasks not evenly distributed across executor/worker nodes:
    - [_Apache JIRA SPARK-19371:_ Cannot spread cached partitions evenly across executors](https://issues.apache.org/jira/browse/SPARK-19371)
        - ___partial temporary resolution__:_ set `spark.locality.wait` config to `0` to make free executor processes
        immediately grab unprocessed data partitions even if such paritions have to be moved/shuffled from other
        executor nodes

- `JVM` 64KB code-gen limit:
    - __RESOLVED__ [_Apache JIRA SPARK-8443:_ GenerateMutableProjection Exceeds JVM Code Size Limits](https://issues.apache.org/jira/browse/SPARK-8443)
    - https://issues.apache.org/jira/browse/SPARK-14554
    - https://issues.apache.org/jira/browse/SPARK-16845
    - https://issues.apache.org/jira/browse/SPARK-17092
    - https://issues.apache.org/jira/browse/SPARK-18207
    - http://stackoverflow.com/questions/6570343/maximum-size-of-a-method-in-java
    - http://stackoverflow.com/questions/17422480/maximum-size-of-a-method-in-java-7-and-8
    - http://stackoverflow.com/questions/32201074/java-how-to-overcome-the-maximum-method-size-in-automatically-generated-code
    - https://coderanch.com/t/482213/java/fix-java-method-size-limit
    - https://www.mail-archive.com/commits@spark.apache.org/msg16036.html
    - https://github.com/apache/spark/pull/15480   CLOSEST POSSIBLE SOLUTION?


- Exploding ``Spark`` graph:
    - http://apache-spark-user-list.1001560.n3.nabble.com/Spark-app-gets-slower-as-it-gets-executed-more-times-td1089.html
    - http://stackoverflow.com/questions/31659404/spark-iteration-time-increasing-exponentially-when-using-join
    - http://stackoverflow.com/questions/32349611/what-should-be-the-optimal-value-for-spark-sql-shuffle-partitions-or-how-do-we-i
    - http://stackoverflow.com/questions/34461804/stackoverflow-due-to-long-rdd-lineage
    - http://stackoverflow.com/questions/38417441/pyspark-socket-timeout-exception-after-application-running-for-a-while
    - http://stackoverflow.com/questions/39084739/evaluating-spark-dataframe-in-loop-slows-down-with-every-iteration-all-work-don
    



### 1. Resolved Issues to Note

- ``Spark` driver requires large memory space for serialized results even there are no data collected to the driver:
    - https://issues.apache.org/jira/browse/SPARK-12837


- `Spark` sampling methods:
    - https://github.com/apache/spark/pull/17141 [SPARK-19800][SS][WIP] Implement one kind of streaming sampling - reservoir sampling
    - Reservoir Sampling: https://spark.apache.org/docs/1.2.0/api/java/index.html?org/apache/spark/util/random/SamplingUtils.html
    

- ``PySpark`` 3-second socket timeout implemented for ``RDD.toLocalIterator()``:
    - https://issues.apache.org/jira/browse/SPARK-18281


- `Spark ML` StackOverflow errors:
    - https://stackoverflow.com/questions/42542875/apache-spark-nondeterministic-stackoverflowerror-when-training-with-gbtregresso

 
- Regression API bugs:
    - https://issues.apache.org/jira/browse/SPARK-17508 Setting weightCol to None in ML library causes an error


- Spark joins not preserving ordering:
    - http://stackoverflow.com/questions/38085801/can-dataframe-joins-in-spark-preserve-order


- Spark checkpointing -- things to note:
    - http://stackoverflow.com/questions/35127720/what-is-the-difference-between-spark-checkpoint-and-persist-to-a-disk




``Keras``:

- https://blog.keras.io/introducing-keras-2.html `Keras` 2.x announcement 
- https://github.com/fchollet/keras/issues/1638   Proper way of making a data generator which can handle multiple workers
- https://github.com/fchollet/keras/issues/4142   using fit_generator with nb_worker > 1 and pickle_safe=True
- https://github.com/tensorflow/tensorflow/issues/8787   Integrate Keras with TFRecords
- http://stackoverflow.com/questions/42184863/how-do-you-make-tensorflow-keras-fast-with-a-tfrecord-dataset


``Python`` packaging

- http://stackoverflow.com/questions/12518499/pip-ignores-dependency-links-in-setup-py   pip ignores dependency_links in setup.py
- http://stackoverflow.com/questions/16584552/how-to-state-in-requirements-txt-a-direct-github-source   How to state in requirements.txt a direct github source


`Sphinx` documentation:

- https://github.com/sphinx-doc/sphinx/issues/628 autoclass member-order doesn't apply to inherited-members
