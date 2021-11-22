"""
    spark is a unified analytic engine for large-scale distributed data processing. 100x faster than hadoop mapreduce.
        support java, python, scala, R, SQL. Spark combine Spark SQL, Spark streaming, MLlib, GraphX
    PySpark is a Python API for Apache Spark.
        In-memory computation, Distributed processing using parallelize, Can be used with many cluster managers (Spark,
        Yarn, Mesos e.t.c), Fault-tolerant, Immutable, Lazy evaluation, Cache & persistence, Inbuild-optimization when
        using DataFrames, Supports ANSI SQL
    Apache Spark works in a master-slave architecture where the master is called “Driver” and slaves are called
        “Workers”. When you run a Spark application, Spark Driver creates a context that is an entry point to your
        application, and all operations (transformations and actions) are executed on worker nodes, and the resources
        are managed by Cluster Manager (standalone, apache mesos, Hadoop YARN, Kubernetes).
    
    SparkSession
        an entry point to work with RDD, DataFrame since Spark 2.0. it's a combined class for all different contexts
        (SQLContext, Streaming Context, Spark Context and HiveContext e.t.c).
        SparkContext is used prior to 2.0 to create Spark RDD, accumulators, and broadcast variables on cluster

    PySpark RDD (Resilient Distributed Dataset) is a fundamental data structure of PySpark that is fault-tolerant,
        immutable distributed collections of objects, which means once you create an RDD you cannot change it. Each
        dataset in RDD is divided into logical partitions, which can be computed on different nodes of the cluster.
        RDD Transformations are lazy operations, only executed after action taken.
        map(), filter(), union(), flatMap(),mapPartition() are narrow transformation (no data sent across partition)
        groupByKey(), reduceByKey(), aggregateByKey(), aggregate(), join(), repartition() are wider(shuffle)
            transformation (data sent across partition), expensive due to shuffling
    
    DataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table 
        in a relational database. mostly similar to Pandas DataFrame with exception PySpark DataFrames are distributed 
        in the cluster
        
    check jobs at Spark Web UI     terminal:  pyspark       localhost:
    check logs at Spark History Server  (https://sparkbyexamples.com/pyspark-tutorial/)

    databricks tools

    installation
        documentation: https://sparkbyexamples.com/pyspark-tutorial/
        install hadoop https://towardsdatascience.com/installing-hadoop-3-2-1-single-node-cluster-on-windows-10-ac258dd48aef

    pip install pyspark
    import pyspark
    from pyspark.sql import SparkSession

    # Spark session
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate() 
        # create spark session
        # master() - specify cluster master  "yarn" or "mesos" for cluster mode, local[x] for Standalone mode
            #  x is number partitions created when using RDD, DataFrame, and Dataset. ideally should be same 
            # as CPU cores 
        # appName() - set your application name.
        # getOrCreate() – returns SparkSession object if exists, creates new one if not

    spark = SparkSession.newSession  # this alternative method always create new session

    common Spark session method
        version() – Returns Spark version
        createDataFrame() – This creates a DataFrame from a collection and an RDD
        getActiveSession() – returns an active Spark session.
        read() – Returns an instance of DataFrameReader class, used to read records file into DataFrame.
        readStream() – Returns an instance of DataStreamReader class, read streaming data into DataFrame.
        sparkContext() – Returns a SparkContext.
        sql() – Returns a DataFrame after executing the SQL.
        sqlContext() – Returns SQLContext.
        stop() – Stop the current SparkContext.
        table() – Returns a DataFrame of a table or view.
        udf() – Creates a PySpark UDF to use it on DataFrame, Dataset, and SQL.


    # Spark Context
        spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
        sparkContext=spark.sparkContext
        sparkContext.stop()  # stop context

    # create RDD

    rdd1 = spark.sparkContext.parallelize([("Java", 20000), ("Python", 100000), ("Scala", 3000)], 2)
        # create RDD from list, 2 is partition number
        # loads the existing collection from your driver program into parallelling RDD
    rdd1 = spark.sparkContext.parallelize([Row(name="James,,Smith",lang=["Java","Scala","C++"],state="CA")])
    rdd2 = spark.sparkContext.textFile("../resources/data/numpy_data.txt")  # Create RDD from external Data source

    # create empty RDD
        rdd = sparkContext.emptyRDD()
        print(rdd.isEmpty())
        rdd = sparkContext.parallelize([])  # default create same partitions as core number add ,10 to override it
        print(str(rdd.getNumPartitions()))
        rdd = df4.rdd    # convert dataframe to rdd

    # perform transformation(return another RDD) and action operations(trigger computation and return RDD values to the 
        # driver(master)). Any operation you perform on RDD runs in parallel.    
    
    # transformation: flatMap(), map(), reduceByKey(), filter(), sortByKey()...
    # action: count(), collect(), first(), max(), reduce()...

    # transformations
        rdd1 = rdd.flatMap(lambda x: x.split(" "))   # flattens the RDD after applying the function and returns a new
                                                      # RDD, here create RDD with single words
        rdd3 = rdd2.map(lambda x: (x,1))   # apply any complex operations like adding a column, updating a column
                                            # here create RDD with key x, value 1 pairs
        rdd5 = rdd4.reduceByKey(lambda a,b: a+b) # merges the values for each key with the function specified.
                                            # here create RDD for appearance counts for unique word
        rdd6 = rdd5.map(lambda x: (x[1],x[0])).sortByKey()     # sort RDD elements on key.
                                            # here switch key with value, and sort via unique words appearance count
        rdd7 = rdd6.filter(lambda x : 'a' in x[1])     #  filter the records in an RDD.
        print(rdd7.collect())            # return list of key-value pair (collect is action)

        rdd.repartition(4)   # increase or decrease the partitions.   change partition number to 4  by moving data from
                             # all partitions (expensive operation).
        rdd1.coalesce(4)     # reduce the number of partitions, movement of the data is lower than repartition

        cache(): caches the RDD
        mapPartitions(): map on each partition
        mapPartitionsWithIndex(): map on each partition with partition index
        randomSplit(): Splits the RDD by the weights specified in the argument  (0.7,0.3)
        union(): combines elements from source dataset and the argument and returns combined dataset
        sample(): returns the sample dataset
        intersection(): returns the dataset which contains elements in both source dataset and an argument
        distinct(): returns the dataset by eliminating all duplicated elements.


    # actions
        counts = rdd6.count()    # Returns the number of records in an RDD
            counts = rdd6.countApprox(1200)    # Returns approximate counts in dataset, return incomplete if timeout
            counts = rdd6.countApproxDistinct()  # Returns approximate unique counts
            counts = rdd6.countByValue()    # return key (unique data value) value (count) pair
        firstRec = rdd6.first()    # Returns the first record.
        recs = rdd6.top(5)     # Returns the first n record.
        datMax = rdd6.max()    # Returns max record.  # min
        totalWordCount = rdd6.reduce(lambda a,b: (a[0]+b[0],a[1]))    # Reduces the records to single
        data3 = rdd6.take(3)    # Returns the first 3 elements of data
            data3 = rdd6.takeOrdered(3)  # Returns the smallest 3 elements of data, load all data to memory
            data3 = rdd6.takeSample(3)  # Returns 3 random elements of data, load all data to memory
        data = rdd6.collect()   # Returns all data from RDD as an array
        rdd6.saveAsTextFile("/tmp/wordCount")    # write the RDD to a text file

        seqOp, combOp = (lambda x, y: x + y), (lambda x, y: x + y)
        agg = rdd.aggregate(0, seqOp, combOp) # Aggregate in each partition(seqOp), then combine all partitions(combOp)
                                              # with initial value 0
        agg = rdd.treeAggregate(0, seqOp, combOp, depth=2)  # Aggregates in a multi-level tree pattern.

        from operator import add
        res = rdd.fold(0, add)   # Aggregate the elements of each partition, then all partitions.
        res = rdd.reduce(add)    # Reduces the elements of the RDD using the specified binary operator
        res = rdd.treeReduce(add)  # Reduces the elements in a multi-level tree pattern.





    
    # Data Frame
    
    # create dataframe
    from pyspark.sql.types import StructType,StructField, StringType, IntegerType
    schema = ['id', 'dogs', 'cats']
    data = [(1, 2, 0), (2, 0, 1)]
    rdd = spark.sparkContext.parallelize(data)
    df = rdd.toDF(schema)
    
        # df = spark.createDataFrame(rdd).toDF(*columns)    # toDF('col1','col2')
        # df = spark.createDataFrame(data=data, schema = schema)
    df.show()

    # schema
           # schema = StructType([StructField('name', StructType([StructField('firstname', StringType(), True),
            StructField('lastname', StringType(), True)])), StructFiled('id'),IntegerType(), True), StructField
            ('hobbies', ArrayType(StringType()), True),StructField('prop', MapType(StringType(),StringType()), True)])

            schema = StructType.fromJson(json.loads(schema.json))  # load schema from json
            print(df.schema.json())  # return dataframe schema json object

            df.schema.fieldNames.contains("firstname")         # check schema exist field
            df.schema.contains(StructField("firstname",StringType,true))

    create dataframe from external csv, text, Avro, Parquet, tsv, xml file
    # create dataframe via read csv
    df = spark.read.csv('../resources/data/Iris.csv')   # return pyspark.sql.dataframe.DataFrame, won't able to process
            # column name (default column name _c0, _c1, first column name row will be treated as data )
            spark.read.csv('./resources/data/Iris.csv', header=True, inferSchema=True)
    df.show()  # print dataframe
        # df.show(self, n=20, truncate=True, vertical=False)   # default data value show 20 characters, use
                # truncate=False to show full or truncate=30 to show 30 char, n=20 show 20 rows

    spark.read.option('header','true').csv('../resources/data/Iris.csv')
    df = spark.read.option('header', 'true').csv('../resources/data/Iris.csv', inferSchema=True)
        # return pyspark.sql.dataframe.DataFrame (data structure), able to treat first row as column name, default
        # datatype is string, add inferSchema=True to change datatype

    df2 = spark.read.text("/src/resources/file.txt")
    df2 = spark.read.json("/src/resources/file.json")

    df.show()

    # some functions and attributes
    print(df.head(3))   # return top 3 rolls as list of pyspark.sql.types.Row
    df.printSchema()   # print structure, column name, type, nullable
    df.dtype()       # return list of tuple [('column name', 'dtype')]
    df.describe()    # return DataFrame with column:dtype inside
    df.describe().show()   # return statistic information of each column
    df.count()          # return rows count
    df.columns          # return list of columns
    df.toPandas()       # convert pySpark dataframe to pandas dataframe, may result memory leak


    #create row
    from pyspark.sql import Row
    row = Row("James",40)              row=Row(name="Alice", age=11)
    print(row[0] +","+str(row[1]))     print(row.name)

    Person = Row("name", "age")  # create a Row like class
    p1=Person("James", 40)
    print(p1.name +","+p2.name)

    # use row to create df
    rdd = spark.sparkContext.parallelize([Row(name="James",lang=["Java","Scala"]), prop=Row(hair="grey",eye="black"))])
    for row in rdd.collect():
        print(row.name, str(row.lang))

    # add row
    newRow = spark.createDataFrame([(4,5,7)], columns)
    appended = df.union(newRow)

    # get row
    dataframe.collect()[0]    # return pyspark.sql.types.Row   get row via index
    dataframe.show(2)    # print top 2 rows
    df.head(2)  or df.take(2)     #  return top 2 rows as list of pyspark.sql.types.Row
    df.tail(2)              #  return last 2 rows as list of pyspark.sql.types.Row
    df = df.limit(5)     # return top 5 rows as dataframe

    # get cell value
    df.select('Id').collect()[1][0]   # get value in 2nd row first column

    # get column
    df.select('Species')   # return a column as DataFrame, can't select by  df['Species']
    df.select('Species').show()   # print all values in a column
    df.select(['Id', 'Species'])   # return two columns as DataFrame
        df.select('Id', 'Species')   # same as above
    df['Species']   or df.Species   # return column object

    # add column
    df.withColumn('new_column', np.random.randn(df.count()))
        df2.withColumn("OtherInfo", struct(col("Id").alias("identifier"))  # copy id to OtherInfo-identifier column

    # drop column
    df = df.drop('new_column')         # drop one column
    df = df.drop(['Species','new_column'])   # drop columns


    # rename column name
    df = df.withColumnRenamed('Id','new Id')   # change column name from Id to new Id

    drop/fill na
    df = df.na.drop()             # drop null rows, default how='any', drop row if any cell is null,
                                  # how='all' need one row with all null value cells to drop
        df = df.na.drop(how='any', thresh=2)  # keep if at least 2 or more cell is not null
        df = df.na.drop(how='any', subset=['Id'])  # drop if any column in subset has null value
    df = df.na.fill(0,['Species','Id'])      # value, subset, fill null with value in subset columns


    # DataFrame also has operations like Transformations and Actions.
    
    



    from pyspark.ml.feature import Imputer
    imputer = Imputer(inputCols=['Id'], outputCols=["{}_imputed".format(c) for c in ['Id']).setStrategy("mean")
            # create column:  col_name_imputed for every column inside inputCols. fill null with mean.  "median"
    df = imputer.fit(df).transform(df)

    
    # filter
    df.filter("SepalLengthCm<=5")     # return dataframe of rows with SepalLengthCm<=5
        # df.filter(df['SepalLengthCm']<=5)   same as above
    df.filter((df['SepalLengthCm']==5) & (df['Id']>=5))    # return dataframe of rows with multiple conditions   &  |  ~

    # group by
    df2 =df.groupby('Species')   # return pyspark.sql.group.GroupedData
    df2 =df.groupby('Species').avg()    # return a dataframe with column species and columns able to take average(
                                    # numbers dtype) with avg(col_name) as column name.    count max mean min sum
    df2 = df.agg({'Species':'mean'})
    
    
    
    # PySpark SQL 
    PySpark SQL is used for processing structured columnar data format. interact with the DataFramedata using SQL syntax.
    
        spark = SparkSession.builder.appName('practice2').getOrCreate()
        df = spark.read.csv('./resources/data/Iris.csv', header=True, inferSchema=True)
        
        df.createOrReplaceTempView("IRIS")  #  create a temporary table
        df = spark.sql("SELECT * from IRIS")
        groupDF = spark.sql("SELECT Species, count(*), avg(SepalWidthCm) from IRIS group by Species")
        groupDF.show()
      
        
      
    # PySpark Streaming
    PySpark Streaming is a scalable, high-throughput, fault-tolerant streaming processing system that supports both batch 
        and streaming workloads. It process real-time data from: file system folder, TCP socket, S3, Kafka, Flume, Twitter,
        and Amazon Kinesis e.t.c. Processed data can be pushed to databases, Kafka, live dashboards e.t.c
        
        # TCP socket
        df = spark.readStream.format("socket").option("host","localhost").option("port","9090").load()
        # write to terminal
        query = df.writeStream.format("console").outputMode("complete").start().awaitTermination()
        
        #Kafka
        df = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "192.168.1.100:9092")
            .option("subscribe", "json_topic").option("startingOffsets", "earliest").load() // From starting
        df.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(*)) AS value").writeStream.format("kafka")
            .outputMode("append").option("kafka.bootstrap.servers", "192.168.1.100:9092").option("topic", "josn_data_topic")
            .start().awaitTermination()    
    
    # PySpark GraphFrames


    # shuffles
    PySpark shuffling triggers when we perform certain transformation operations like groupByKey(), reduceByKey(),
        countByKey(), join() on RDDS. It is an expensive operation (disk I/O, serialization, network I/O)
    Shuffle partition size: if small dataset, use less partition otherwise not enough data. for large dataset, use
        more partitions otherwise out of memory

    # cache and persist
        Using cache() and persist() methods, Spark provides an optimization mechanism to store the intermediate
        computation of an RDD, DataFrame, and Dataset.cache() default saves it to memory (MEMORY_ONLY), persist() store
        at user-defined storage level.
        df.cache()  or rdd.cache()
        dfPersist = rdd.persist()  # default MEMORY_AND_DISK_DESER  #  MEMORY_ONLY, MEMORY_AND_DISK, MEMORY_ONLY_SER,
                #MEMORY_AND_DISK_SER, DISK_ONLY,MEMORY_ONLY_2, MEMORY_AND_DISK_2  (serialized, replicate 2 nodes)
        rdd.persist(StorageLevel.MEMORY_AND_DISK).is_cached
        rddPersist2 = rddPersist.unpersist()  # manually remove from persist, default drops persisted data if not used
                                              # or by using least-recently-used (LRU) algorithm.


    # Shared Variables
        Broadcast variables (read-only shared variable),   Accumulator variables (updatable shared variables)
        Broadcast variables are read-only shared variables that are cached and available on all nodes in a cluster
        in-order to access or use by the tasks. Instead of sending this data along with every task, PySpark distributes
        broadcast variables to the machine using efficient broadcast algorithms to reduce communication costs.

        # Broadcast
        broadcastVar = sc.broadcast({"NY":"New York", "CA":"California", "FL":"Florida"})   # sc is sparkcontext
        val = broadcastVar.value['BY']  # New York, read only variable cached and available on all nodes

        PySpark Accumulators are only “added” through an associative and commutative operation and are used to perform
            counters (Similar to Map-reduce counters) or sum operations.

        # Accumulator
        accum = sc.accumulator(0)   # shared variable initial value 0, use int and float, LongAccumulator,
                #  DoubleAccumulator, and CollectionAccumulator, use AccumulatprParam for custom data type
        rdd = spark.sparkContext.parallelize([1,2,3,4,5])
        rdd.foreach(lambda x: accum.add(x))    #  add/update a value in accumulator
            # def countFun(x):
            #     global accuSum
            #     accuSum += x
            # rdd.foreach(countFun)
        print(accum.value)   # Accessed by driver to retrieve the accumulator value

"""
import pyspark
from pyspark.sql import SparkSession
import numpy as np
from pyspark import SparkContext
import os
def test():

    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    sparkContext=spark.sparkContext
    rdd=sparkContext.parallelize([1,2,3,4,5], 2)
    print(rdd.count())
def test1():
    spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

    states = {"NY": "New York", "CA": "California", "FL": "Florida"}
    broadcastStates = spark.sparkContext.broadcast(states)

    data = [("James", "Smith", "USA", "CA"),
            ("Michael", "Rose", "USA", "NY"),
            ("Robert", "Williams", "USA", "CA"),
            ("Maria", "Jones", "USA", "FL")
            ]

    columns = ["firstname", "lastname", "country", "state"]
    df = spark.createDataFrame(data=data, schema=columns)
    df.printSchema()
    df.show(truncate=False)

    def state_convert(code):
        return broadcastStates.value[code]

    result = df.rdd.map(lambda x: (x[0], x[1], x[2], state_convert(x[3]))).toDF(columns)
    result.show(truncate=False)

def test2():
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate() 
        # create spark session
        # master() - specify cluster master  "yarn" or "mesos" for cluster mode, local[x] for Standalone mode
            #  x is number partitions created when using RDD, DataFrame, and Dataset. ideally should be same 
            # as CPU cores 
        # appName() - set your application name.
        # getOrCreate() – returns SparkSession object if exists, creates new one if not
    
    # create RDD 
    rdd1 = spark.sparkContext.parallelize([("Java", 20000), ("Python", 100000), ("Scala", 3000)])   # create RDD from list
        # SparkContext is entry point to Spark to create Spark RDD, accumulators, and broadcast variables on cluster
    rdd2 = spark.sparkContext.textFile("../resources/data/numpy_data.txt")  # Create RDD from external Data source
    
    # perform transformation(return another RDD) and action operations(trigger computation and return RDD values to the 
        # driver(master)). Any operation you perform on RDD runs in parallel.    
    
    # transformation: flatMap(), map(), reduceByKey(), filter(), sortByKey()...
    # action: count(), collect(), first(), max(), reduce()...
    counts = rdd1.count()  # 3
    print("Number of elements in RDD -> %i" % counts)
    
    
    spark.sparkContext.stop()  # stop context
    
def basic():

    spark = SparkSession.builder.appName('practice').getOrCreate()  # session name
    df = spark.read.csv('../resources/data/Iris.csv', header=True, inferSchema=True)
        # return DataFrame, won't able to process column name (default
            # column name _c0, _c1, first column name row will be treated as data )
    df.show()  # print dataframe
    print(type(df))
    df = spark.read.option('header', 'true').csv('../resources/data/Iris.csv', inferSchema=True)
    df.show()
    print(type(df))
    print(type(df.head(3)[0]))
    df.printSchema()
    df.select('Species').show()
    #df.count()
    print(df.columns)
    df = df.withColumn('new_column', df.Id * 2)
    df = df.drop('new_column')
    df = df.withColumnRenamed('Id','new Id')
    df.show()
    df2 = df.groupby('Species').avg()
    df2.show()

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
def ml():
    spark = SparkSession.builder.appName('practice').getOrCreate()  # session name
    df = spark.read.csv('../resources/data/Iris.csv', header=True, inferSchema=True)
    featureAssembler = VectorAssembler(inputCols=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',],
                                       outputCol='Independent Feature')  # group features columns into one column
    data = featureAssembler.transform(df)   # return dataframe with extra Independent Feature column


    # regression
    train_data, test_data = data.randomSplit([0.75, 0.25])  # train test split
    regressor = LinearRegression(featuresCol='Independent Feature',labelCol='SpeciesIndex')
    regressor.fit(train_data)
    print(regressor.coefficient, regressor.intercept)   # .coefficients, .intercept
    pred = regressor.evaluate(test_data)
    pred.predictions.show()
    print(pred.meanAbsoluteError, pred.meanSquaredError, pred.r2)

    # classification
    indexer = StringIndexer(inputCol="Species", outputCol="SpeciesIndex")
        # add SpeciesIndex column with 0,1,2... for unique value
        # indexer = StringIndexer(inputCols=["Species","Gender"], outputCol=["Species_index","Gender_index"])
            # can be multiple column with list input
    data = indexer.fit(data).transform(data)
    data = data.select('Independent Feature', 'SpeciesIndex')
    data.show(100)

    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
            # ,family="multinomial" for binary classification

    # Fit the model
    lrModel = lr.fit(train_data)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    trainingSummary = lrModel.summary

    # Obtain the objective per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("objectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold
    maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
    bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']
    lr.setThreshold(bestThreshold)


if __name__ == '__main__':
    test1()
    #test()
    #basic()
    #ml()
