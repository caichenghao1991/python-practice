"""
    spark is a unified analytic engine for large-scale distributed data processing. 100x faster than hadoop mapreduce.
        support java, python, scala, R, SQL. Spark combine Spark SQL, Spark streaming, MLlib, GraphX
    PySpark is a Python API for Apache Spark.
        In-memory computation, Distributed processing using parallelize, Can be used with many cluster managers (Spark,
        Yarn, Mesos e.t.c), Fault-tolerant, Immutable, Lazy evaluation, Cache & persistence, Inbuild-optimization when
        using DataFrames, Supports ANSI SQL
    Apache Spark works in a master-slave architecture where the master is called “Driver” and slaves are called
        “Workers”. When you run a Spark application, Spark Driver creates a context that is an entry point to your
        application, and all  operations (transformations and actions) are executed on worker nodes, and the resources
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
        # .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/test.coll").config("spark.mongodb.output.uri",
            "mongodb://127.0.0.1/test.coll")   # connect to mongodb

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
    rdd1 = spark.sparkContext.parallelize([Row(name="James,Smith",lang=["Java","Scala","C++"],state="CA")])
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
            rdd.sample(False,0.1,0)       # sample(self, withReplacement, fraction, seed=None)
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
            seqOp = (lambda x, y: (x[0] + y, x[1] + 1)) or udf
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
                # whether can be null
            schema = StructType.fromJson(json.loads(schema.json))  # load schema from json
            print(df.schema.json())  # return dataframe schema json object

            df.schema.fieldNames.contains("firstname")         # check schema exist field
            df.schema.contains(StructField("firstname",StringType,true))

    create dataframe from external csv, text, Avro, Parquet, tsv, xml file
    # create dataframe via read csv
    df = spark.read.csv('../resources/data/Iris.csv')   # return pyspark.sql.dataframe.DataFrame, won't able to process
            # column name (default column name _c0, _c1, first column name row will be treated as data )
            spark.read.csv('./resources/data/Iris.csv','b.csv', header=True, inferSchema=True)
            or df = spark.read.format("csv").load('a.csv')

    df.show()  # print dataframe
        # df.show(self, n=20, truncate=True, vertical=False)   # default data value show 20 characters, use
                # truncate=False to show full or truncate=30 to show 30 char, n=20 show 20 rows

    spark.read.option('header','true').csv('../resources/data/Iris.csv')
    df = spark.read.option('header', 'true').csv('../resources/data/Iris.csv', inferSchema=True)
        # return pyspark.sql.dataframe.DataFrame (data structure), able to treat first row as column name, default
        # datatype is string, add inferSchema=True to change datatype
        # .options(header='True', inferSchema='True',delimiter=',')  specify file delimiter
        # .schema(schema).load('a.csv')  # specify schema

    df2.write.format("csv").save("/tmp/spark_output/zipcodes")
    df2.write.mode('overwrite').options(header='True', delimiter=',').csv("/tmp/spark_output/zipcodes")
        # overwrite, append, ignore(ignore file exist), error (default, error when file exist)

    df2 = spark.read.text("/src/resources/file.txt")

    df2 = spark.read.schema(schema).json(["/src/resources/file.json","resources/*.json"])
        # df = spark.read.option("multiline","true").format('org.apache.spark.sql.json').load("resources/zipcodes.json")
    df2.write.mode('Overwrite').json("/src/resources/file.txt")

    df = spark.read.parquet("/temp/out/people.parquet")
    df = df.write.parquet("/tmp/out/people.parquet")

    df = spark.read.format("mongo").load()
        # df = spark.read.format("mongo").option("uri", "mongodb://127.0.0.1/people.contacts").load()
    df.write.format("mongo").mode("append").save()


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

    # use row to create rdd
    rdd = spark.sparkContext.parallelize([Row(name="James",lang=["Java","Scala"]), prop=Row(hair="grey",eye="black"))])

    # print row element value
    for row in rdd.collect():
        print(row.name, str(row.lang),row['name'],row[0])

    # add row
    newRow = spark.createDataFrame([(4,5,7)], columns)
    appended = df.union(newRow)

    # get row
    dataframe.collect()[0]    # return pyspark.sql.types.Row   get row via index
    dataframe.show(2)    # print top 2 rows
    df.head(2)  or df.take(2)     #  return top 2 rows as list of pyspark.sql.types.Row
    df.tail(2)              #  return last 2 rows as list of pyspark.sql.types.Row
    df = df.limit(5)     # return top 5 rows as dataframe

    # Columns
    # create column object
    from pyspark.sql.functions import col
    colObj = lit('Id')  or col('Id')
    df.select(colObj, col('name'), 'gender', col("`name.fname`"), "`name.fname`", df.prop.hair, df["prop.hair"],
        col("prop.hair"), col("prop.*"))

    df = df.select(df.col1 + df.col2).show()   # -, *, /, %, >, <, ==

    # get cell value
    df.select('Id').collect()[1][0]   # get value in 2nd row first column

    # get column
    df.select('Species')   # return a column as DataFrame, can't select by  df['Species']
    df.select('Species').show()   # print all values in a column
    df.select(['Id', 'Species'])   # return two columns as DataFrame
        df.select('Id', 'Species')   # same as above
    df['Species']   or df.Species   # return column object

    # column functions
    alias(), name():   df.select(df.fname.alias("first_name"))   # Provides alias to the column or expressions
    asc(): df.sort(df.fname.asc()).show()   # Returns ascending order of the column.
    astype(), cast(): df.select(df.id.cast("int"))   # cast the data type to another type
    between():  df.filter(df.id.between(100,300))   # returns boolean: columns values are between lower and upper bound
    contains():  df.filter(df.fname.contains("Cruise"))   # Check if String contains in another string.
    startswith(), endswith():  df.filter(df.fname.startswith("T"))  # returns boolean: String starts/ends with.
    isNotNull(), isNull():  df.filter(df.lname.isNull())   # returns True if the current expression is (NOT) null.
    like(), rlike():  df.filter(df.fname.like("%om"))   # Similar to SQL like(rlike: like with regex) expression.
    substr():  df.select(df.fname.substr(1,2))   # return a Column which is a substring of the column.
    when(), otherwise():  df.select(when(df.gender=="M","Male").otherwise(df.gender=="Female"))   # Similar to SQL CASE
                        # WHEN, Executes a list of conditions and returns one of multiple possible result expressions.
    isin():  df.filter(df.id.isin(['1','2']))  # return boolean: check if expression is contained by arguments
    getField():  df.select(df.properties.getField("hair"))  # returns a field by name in a StructField and by key in Map
    getItem():  df.select(df.properties.getItem("hair"))   # returns a values from Map/Key at the provided position.
    asc_null_first(), asc_null_last(), desc_null_first(), desc_null_last(): Returns (de)ascending order null first/last
    bitwiseAND(), bitwiseOR(), bitwiseXOR():  Compute bitwise AND, OR & XOR of this expression with arguments
    eqNullSafe(): equality test that is safe for null values.
    over(): used with window column
    dropFields(): used to drops fields in StructType by name
    withFields(fieldName, col): adds/replaces a field in StructType by name


    # withcolumn
    df.withColumn('new_column', np.random.randn(df.count()))   # add column
    df.withColumn("Country", lit("USA"))  # add column
        df2.withColumn("OtherInfo", struct(col("Id").alias("identifier"))  # copy id to OtherInfo-identifier column
    df.withColumn("salary",col("salary").cast("Integer"))  # cast to datatye
    df.withColumn("salary",col("salary")*100)   # update column value


    # drop column
    df = df.drop('new_column')         # drop one column
    df = df.drop(['Species','new_column'])   # drop columns

    # select column  (transformation)
    colObj = lit('Id')  or col('Id')
    df.select(colObj, col('name'), 'gender', col("`name.fname`"), "`name.fname`", "name.fname" df.prop.hair,
        df["prop.hair"], col("prop.hair"), col("prop.*"),"prop.*")
    df.select(df.colRegex("`^.*name*`")).show()   # regex
    df.select([col for col in df.columns])       # select all columns
    df.select("*")
    df.select(df.columns[2:4])


    # rename column name
    df = df.withColumnRenamed('Id','new Id')   # change column name from Id to new Id

    schema = StructType([StructField("fname",StringType()),StructField("lname",StringType())])
    df.select(col("name").cast(schema2), col("dob"),col("salary").alias("income")).withColumn("income",col("salary"))

    newColumns = ["newCol1","newCol2","newCol3","newCol4"]
    df.toDF(*newColumns)     # rename only flat schema dataframe


    # drop/fill na
    df = df.na.drop()             # drop null rows, default how='any', drop row if any cell is null,
                                  # how='all' need one row with all null value cells to drop
        df = df.na.drop(how='any', thresh=2)  # keep if at least 2 or more cell is not null
        df = df.na.drop(how='any', subset=['Id'])  # drop if any column in subset has null value
    df = df.na.fill(0,['Species','Id'])      # value, subset, fill null with value in subset columns
    df = df.na.fill("").na.fill("",['Id'])  # fill with empty string
    df = df.na.fill({"city": "unknown", "type": ""})

    # fill na with mean/median
    from pyspark.ml.feature import Imputer
    imputer = Imputer(inputCols=['Id'], outputCols=["{}_imputed".format(c) for c in ['Id']).setStrategy("mean")
            # create column:  col_name_imputed for every column inside inputCols. fill null with mean.  "median"
    df = imputer.fit(df).transform(df)


    # dropDuplicates(), distinct()
    df = df.distinct()     # returns a new DataFrame after removing the duplicate records.
    df.dropDuplicates()     # same as above, based on all column
        # df2 = df.dropDuplicates(["department","salary"])   # drop rows as long as department and salary has same value

    # ArrayType
    from pyspark.sql.types import StringType, ArrayType, MapType
    arrayCol = ArrayType(StringType(),False)   # value type, accept null, a collection of data type
    df.select(df.name,explode(df.languagesAtSchool))  # use explode() to convert array collection to rows
    df.select(split(df.name,",").alias("n_arr"))  # split return ArrayType from split string column
    df.select(array(df.state,df.city).alias("address"))   # create array column by merging other columns
    df.select(df.name,array_contains(df.languages,"Java")   # used to check if array column contains a value. Returns
                                # null if the array is null, true if the array contains the value, and false otherwise.

    # MapType
    represent Python Dictionary (dict) to store key-value pair, a MapType object comprises three fields, keyType
        (a DataType), valueType (a DataType) and valueContainsNull (a BooleanType).
    schema = StructType([StructField('name', StringType(), True),MapType(StringType(),StringType(),True)])
    df = spark.createDataFrame(data=dataDictionary, schema = schema)
    df.withColumn("hair",df.properties.getItem("hair"))
    df.select(df.name,explode(df.properties))   # explode() convert map collection to key, value column
    df.select(df.name,map_keys(df.properties))   # map_keys() return all map keys
    df.select(df.name,map_values(df.properties)).show()  # map_values() return all map values

    # collect
    action operation for retrieve all the elements of the dataset (from all nodes) to the driver node. used for
    smaller dataset usually after filter(), group(). otherwise OutOfMemory error

    dataCollect = deptDF.collect()  # return an array of Row type
    for row in dataCollect:
        print(row['dept_name'],row.name)

    deptDF.collect()[0][0]  # get first element in first row


    # sort()  orderby()  #  default ascending
    df.sort(col("department"),"state")   # sorted by the first department column and then the state column
    df.orderby(col("department").asc(),"state", df.name.desc())

    df.createOrReplaceTempView("EMP")
    spark.sql("select employee_name,department from EMP ORDER BY department asc") # sort raw sql

    # DataFrame also has operations like Transformations and Actions.
    

    # map
        RDD transformation used to apply the transformation function (lambda) on every element of RDD/DataFrame and
            returns a new RDD.
        rdd2 = df.rdd.map(lambda x: (x.name+","+x[1],x[2],x["salary"]*2))    # need to convert to RDD to use map()
        df2 = rdd2.toDF(["name","gender","new_salary"])

        def func(x):  # use custom function
            return (x.name, x.salary*2)
        rdd2 = df.rdd.map(lambda x: func(x))

    # for each
    def f(x): print(x)
    df.foreach(f)
    df.foreach(lambda x: print("x["firstname"])

    for row in df.collect():   # df.rdd.toLocalIterator()
        print(row['firstname'])

    # filter
    df.filter("SepalLengthCm<=5")     # return dataframe of rows with SepalLengthCm<=5
        # df.filter(df['SepalLengthCm']<=5)   same as above
    df.filter((df['SepalLengthCm']==5) & (df['Id']>=5))    # return dataframe of rows with multiple conditions
        ==  !=  <>     &  |  ~
    df.filter(df.state.isin(["OH","CA","DE"]))
    df.filter(df.state.startswith("N"))   # endswith()  contains()
    df2.filter(df2.name.like("%rose%"))   # sql like
    df2.filter(df2.name.rlike("(?i)^*rose$"))   # sql like regex

    from pyspark.sql.functions import array_contains
    df.filter(array_contains(df.languages,"Java"))    # filter rows base on value present in an array collection column

    df.filter(df.name.lastname == "Williams")   # nested Struct columns


    # group by
    collect the identical data into groups on DataFrame and perform aggregate functions on the grouped data.

    df2 =df.groupby('Species')   # return pyspark.sql.group.GroupedData
    df2 =df.groupby('Species').avg('width')    # return a dataframe with column species and columns able to take
                                    #  average(numbers dtype) with avg(col_name) as column name.
        # count(), mean(), max(), min(), sum(), avg(), agg(), pivot()
    df.groupBy("department","state").sum("salary","bonus")   # group by 2 columns and create 2 sum column
    df.groupBy("department").agg(sum("salary").alias("sum_salary"),avg("salary").alias("avg_salary"))
    df2 = df.agg({'Species':'mean'})
    df.groupBy("department").agg(sum("salary").alias("sum_salary")).where(col("sum_salary") >=6000))
    

    # join
    combine two DataFrames and by chaining these you can join multiple DataFrames
    how: inner, cross, outer,full, full_outer, left, left_outer, right, right_outer,left_semi, and left_anti

    df1.join(df2,df1.id ==  df2.id,"inner").join(df3,df1.id ==  df3.id,"inner")

    # union(), unionAll()
    union(): merge two DataFrame’s of the same structure/schema, replace unionAll() after spark 2.0.0
    df = df.union(df2)  # this will merge df and df2, keep duplicates. add .distinct() to remove duplicates

    for column in [column for column in df2.columns if column not in df.columns]:
        df = df.withColumn(column, lit(None))   #add missing column in df1
    df = df.unionByName(df2)  # unionByName() is used to merge two DataFrame’s by column names instead of by rows.


    # RDD.sample,  RDD.takeSample(), DataFrame.sample, DataFrame.sampleBy()
    # sample(): get random sample records from the dataset
    sample(withReplacement=False, fraction, seed=None)   fraction [0.0,1.0]
    df = df.sample(True, 0.06, 8)  # fraction 0.06, seed 8, with replacements

    Stratified(weighted) sampling  in PySpark without replacement
    df = df.sampleBy("gender", {"male": 0.1, "female": 0.2}, 8)   # default weight 0 if not specified, seed 8

    # pivot
    pivot() used to rotate the data from one column into multiple columns. Unpivot is a reverse operation, rotating
        column values into rows values.

    pivotDF = df.groupBy("Product").pivot("Country").sum("Amount")
    # in addition to Product column, convert different value in country into columns with value of sum for each country
    countries = ["USA","China","Canada","Mexico"]  # for performance pre pyspark 2.0
    pivotDF = df.groupBy("Product").pivot("Country", countries).sum("Amount")

    from pyspark.sql.functions import expr
    unpivotExpr = "stack(3, 'Canada', Canada, 'China', China, 'Mexico', Mexico) as (Country,Total)"
    unPivotDF = pivotDF.select("Product", expr(unpivotExpr)).where("Total is not null")  # unpivot

    # user define function (UDF most expensive operations)
    UDF’s are a black box to PySpark hence it can’t apply optimization and you will lose all the optimization
        PySpark does on Dataframe/Dataset
    create a function in a Python syntax and wrap it with PySpark SQL udf() or register it as udf and use it on
        DataFrame and SQL respectively

    def convertCase(str):  # create python function
        resStr=""
        for x in str.split(" "):
            resStr= resStr + x[0:1].upper() + x[1:len(x)] + " "
        return resStr
    convertUDF = udf(lambda z: convertCase(z),StringType())   #convert python function to spark UDF
    df.select(col("Id"), convertUDF(col("Name")))
    df.withColumn("Name", convertUDF(col("Name"))

    @udf(returnType=StringType())
    def upperCase(str):
        return str.upper()
    df.withColumn("Name", upperCase(col("Name")))

    # Using UDF on SQL
    spark.udf.register("convertUDF", convertCase,StringType())
    df.createOrReplaceTempView("NAME_TABLE")
    df = spark.sql("select Id, convertUDF(Name) as Name from NAME_TABLE where Name is not null")
        # need handle null manually otherwise runtime exception


    # partitionBy()
    partition the large dataset (DataFrame) into smaller files based on one or multiple columns while writing to disk,
    having too many partitions creates too many sub-directories on HDFS, brings unnecessarily and overhead to NameNode

    partitionBy() is a function of pyspark.sql.DataFrameWriter class which is used to partition based on column values
        while writing DataFrame to Disk/File system.

    df.write.option("header",True).partitionBy("state").mode("overwrite").csv("/tmp/zipcodes-state")
        # partition to same number of partition as number of state, create one csv for each  with name state=IL,...
        # .partitionBy("state","city")  # for each unique (state, city) create a partition, create state sub directory
        # df.write.option("maxRecordsPerFile", 2)    # config each partition hold 2 records

    df.repartition(4)   # repartition to 2 partitions


    # other aggregate functions
    df.select(approx_count_distinct("salary")   #returns the count of distinct items in a group.
    df.select(avg("salary"))   # returns the average of values in the input column
    df.select(collect_list("salary"))    # returns all values from an input column with duplicates.
    df.select(collect_set("salary"))   # returns all values from an input column with duplicate values eliminated.
    df.select(countDistinct("department", "salary"))    # returns the number of distinct elements in a columns
    df.select(count("salary"))    # returns number of elements in a column.
    df.select(first("salary"))  # returns the first/last element in a column
    max(), min(), mean(), sum(), sumDistinct(),skewness(), stddev(), stddev_samp(), stddev_pop(), variance(), kurtosis()
        var_samp(), var_pop()   # return xxx of the values in a column




    # window function
     operate on a group of rows and return a single value for every input row.
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number
    windowSpec  = Window.partitionBy("department").orderBy("salary")  #  need to partition and order first
    df = df.withColumn("row_number",row_number().over(windowSpec))   # row_number() is the row index for each partition
                                                    # different value for duplicate row
    df = df.withColumn("rank",rank().over(windowSpec))   # rank() is the row index for each partition, with same value
                                                    # for duplicate row, then jump the index number if duplication
    df = df.withColumn("rank",dense_rank().over(windowSpec))   # dense_rank() is the row index for each partition, with
                                                    #  same value for duplicate row, no jump in index if duplication
    df = df.withColumn("percent_rank",percent_rank().over(windowSpec))   # percent_rank() is row percentage for each
                        # partition, same value for duplicate row, no jump in index if duplication
    df = df.withColumn("ntile",ntile(2).over(windowSpec))   # returns the relative rank of result rows within a window
                        # partition. start 1, till x (here 2). same value for duplicate row, no jump in index
    df = df.withColumn("cume_dist",cume_dist().over(windowSpec))     # get the cumulative distribution of values within
                        # a window partition. same value for duplicate row, no jump in index
    df = df.withColumn("avg", avg(col("salary")).over(windowSpecAgg))     # min, max, sum of column salary respect for
                        # different departments
    df.withColumn("lag",lag("salary",2).over(windowSpec))   # shift salary column value down 2, with first 2 fill null
    df.withColumn("lead",lead("salary",2).over(windowSpec))   # shift salary column up 2, with last 2 fill null


    # date and timestamp
    DateType default format is yyyy-MM-dd,  TimestampType default format is yyyy-MM-dd HH:mm:ss.SSSS
    Returns null if the input is a string that can not be cast to Date or Timestamp.

    current_date()	# Returns the current date as a date column.
    date_format(dateExpr,format)	# Converts a date/timestamp/string to a value of string in the format specified by
        # the date format given by the second argument.
    to_date()	# Converts the column into `DateType` by casting rules to `DateType`.
    to_date(column, fmt)	# Converts the column into a `DateType` with a specified format
    add_months(Column, numMonths)	# Returns the date that is `numMonths` after `startDate`.
    date_add(column, days), date_sub(column, days)  # Returns the date that is `days` days before/ after `start`
    datediff(end, start)	# Returns the number of days from `start` to `end`.
    months_between(end, start)	# Returns number of months between dates `start` and `end`. assume 31 days per month
    months_between(end, start, roundOff)  #If roundOff is true, result is rounded off to 8 digits; not rounded otherwise
    next_day(column, dayOfWeek)	 # Returns the first date which is later than the value of the `date` column that is on
        # the specified day of the week.
    trunc(column, format)	# Returns date truncated to the unit specified by the format.
    date_trunc(format, timestamp)	#Returns timestamp truncated to the unit specified by the format.
        # format: year: 'year', 'yyyy', 'yy'; month:'month', 'mon', 'mm'; day:'day', 'dd' to truncate by day;
            # 'second', 'minute', 'hour', 'week', 'month', 'quarter'
    year(column), quarter(column), month(column), dayofweek(column), dayofmonth(column), dayofyear(column),
        weekofyear(column) 	 # Extracts the xxx as an integer from a given date/timestamp/string
    last_day(column)	# Returns the last day of the month which the given date belongs to.
    from_unixtime(column)	# Converts the number of seconds from unix epoch to a string representing the timestamp of
        # that moment in the current system time zone in the yyyy-MM-dd HH:mm:ss format.
    from_unixtime(column, f)	# extra given format.
    unix_timestamp()	# Returns the current Unix timestamp (in seconds) as a long
    unix_timestamp(column)	Converts time string in format yyyy-MM-dd HH:mm:ss to Unix timestamp (in seconds), using the
        # default timezone and the default locale.
    unix_timestamp(column, p)	# Converts time string with given pattern to Unix timestamp (in seconds).

    current_timestamp ()	Returns the current timestamp as a timestamp column
    hour(column), minute(column), second(column)	# Extracts xxx as an integer from a given date/timestamp/string.
    to_timestamp(column)	Converts to a timestamp by casting rules to `TimestampType`.
    to_timestamp(column, fmt)	Converts time string with the given pattern to timestamp.

    from pyspark.sql.functions import *
    df = df.select(current_date())
    df.select(date_format(col("input"), "MM-dd-yyyy"), to_date(col("input"), "yyy-MM-dd"),
        datediff(current_date(),col("input")), months_between(current_date(),col("input")), trunc(col("input"),"Month")
        add_months(col("input"),-3), year(col("input")), dayofweek(col("input")))
    df.select(current_timestamp(), to_timestamp(col("input"), "MM-dd-yyyy HH mm ss SSS"), hour(col("input")))


    # JSON
    jsonString='''{"Zipcode":704,"ZipCodeType":"STANDARD","City":"PARC PARQUE","State":"PR"}'''
    df = spark.createDataFrame([(1, jsonString)],["id","value"])
    df2 = df.withColumn("value",from_json(df.value,MapType(StringType(),StringType()))) # from_json() convert JSON
        # string into Struct type or Map type.
    df2.withColumn("value",to_json(col("value")))  # to_json() convert DataFrame columns MapType or Struct type to JSON
    df.select(col("id"),json_tuple(col("value"),"Zipcode","ZipCodeType","City"))   # json_tuple() query or extract the
        # elements from JSON column and create the result as a new columns.
    df.select(col("id"),get_json_object(col("value"),"$.ZipCodeType"))   # get_json_object() extract the JSON string
        # based on path from the JSON column.
    schemaStr=spark.range(1).select(schema_of_json(lit('''{"Zipcode":704,"ZipCodeType":"STANDARD","City":"PARC PARQUE",
        "State":"PR"}''''))).collect()[0][0]    # schema_of_json() to create schema string from JSON string column.


    # PySpark SQL
    PySpark SQL is used for processing structured columnar data format. interact with the DataFramedata using SQL syntax.

        spark = SparkSession.builder.appName('practice2').getOrCreate()
        df = spark.read.csv('./resources/data/Iris.csv', header=True, inferSchema=True)

        df.createOrReplaceTempView("IRIS")  #  create a temporary table
        df = spark.sql("SELECT * from IRIS")
        df = spark.sql("select * from EMP e, DEPT d where e.emp_dept_id == d.dept_id")
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
        val = broadcastVar.value['NY']  # New York, read only variable cached and available on all nodes

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

    # ML lib
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
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    indexer = StringIndexer(inputCol="Species", outputCol="SpeciesIndex")
        # add SpeciesIndex column with 0,1,2... for unique value
        # indexer = StringIndexer(inputCols=["Species","Gender"], outputCol=["Species_index","Gender_index"])
            # can be multiple column with list input
    data = indexer.fit(data).transform(data)

    for categoricalColumn in categoricalColumns:
        stringIndexer = StringIndexer(inputCol=categoricalColumn, outputCol = categoricalColumn+"Index").
            setHandleInvalid("skip")
        dataset = stringIndexer.fit(dataset).transform(dataset)
        encoder = OneHotEncoder(inputCol=categoricalColumn+"Index", outputCol=categoricalColumn+"classVec")
        dataset = encoder.transform(dataset)

    data = data.select('Independent Feature', 'SpeciesIndex')
    data.show(100)

    lr = LogisticRegression(maxIter=20, regParam=0.3, elasticNetParam=0)
    # cross validation param picking
    paramGrid = (ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.3, 0.5]).addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2])
        .addGrid(lr.maxIter, [10, 20, 50]).addGrid(idf.numFeatures, [10, 100, 1000]).build())
    cv = CrossValidator(estimator=lr,estimatorParamMaps=paramGrid,evaluator=evaluator,numFolds=5)
        #  nb = NaiveBayes(smoothing=1)
        # rf = RandomForestClassifier(labelCol='label',featuresCol='features',numTrees=100,maxDepth=4,maxBins=32)
    cv_model = cv.fit(data)
    predictions = cv_model.transform(data)
    predictions.filter(predictions['prediction'] == 0).select('Descript', 'Category', 'probability', 'label',
        'prediction').orderBy('probability', ascending=False).show(n=10, truncate=30)

    evaluator = MulticlassClassificationEvaluator(predictionCol='prediction')
    print(evaluator.evaluate(predictions))


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


    or read data from data source (mongodb), preprocess with pyspark, df.toPandas() then run distributed training
        (tensorflow).


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

