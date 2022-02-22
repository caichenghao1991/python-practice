"""
    database: data persistence, retrieval and management
    relation database(SQL): need accuracy Oracle/ MySQL / MariaDB (similar to MySQL)   (row-column, 2d table)
        one line: a record    one column: attribute
        ACID
        Atomic: All operations in a transaction succeed or every operation is rolled back. (transaction)
        Consistent: On the completion of a transaction, the database is structurally sound.
            instance consistent: ensure no redundancy via primary keys
            relation consistent: ensure relation holds between instances via foreign keys.
            field consistent: data consistency, via datatype, length, null check, default value, check constraint
        Isolated: Transactions do not contend with one another. Contentious access to data is moderated
            by the database so that transactions appear to run sequentially.
            transactions don't know each other's intermediate state
        Durable: The results of applying a transaction are permanent, even in the presence of failures.


    non-relational database (NoSQL): faster flexible. MongoDB, Redis (key-value pair, cache),
        ElasticSearch (search engine)
        BASE
        Basic Availability: the system guarantees availability.
        Soft-state: the state of the system may change over time, even without input.
        Eventual consistency: the system will become consistent over a period of time, given that the system doesn't
            receive input during that period.
        document (MongoDB(able to create some search index, and other SQL functionalities), ElasticSearch),
        key-value (Redis,value can be any dtype), wide-column (Hbase, store on columns), graph store (Neo4J)



    SQL: structured query language
        DDL data definition language:  create  drop  alter
        DML data manipulation language:  select insert delete update    crud
        DCL data control language:   grant  revoke

        can only put one item in a cell, create a new table for list(one to many) item, while non-relational database
            can put a list in a cell



    rpm -ivh mysql-5.7.25-1.el7.x86_64.rpm    # install MySQL
    rpm -e mysql-5.7.25-1  # uninstall mysql

    yum install -y mariadb mariadb-server  # install MariaDB
    systemctl start/stop/status mariadb  # start maria server on centos 8
        # windows  inside bin    mariadbd.exe --console
    net start/stop mysql
    netstat -nap | grep 3306   # maria server default port 3306, open 3306 to specific ip, not to public
        process name mysqld: daemon (won't stop system shutdown, close app same time)

    mysql -u root -p   # first time login to mysql client no need password
    use mysql          # change to mysql table
    update user set password=password('123456') where user='root';
    flush privileges;   # make changes in effect
    quit               # exit mariadb


    Sequel Pro  # visual tool for database
    -- comment in MySQL
    select version()   # show mariadb version
    ? data types;      # show document
    show databases;    # show all databases
    create database university default charset utf8 collate utf8_general_ci;     # utf8mb4  emoji
    drop database if exists university;   drop database university;
    use university;        # switch to university database context
    show tables;       # show tables in the database

    drop table if exists t_student;   # need drop dependent table first
    create table t_student   -- create table student
    (   stu_id int not null,
        stu_name varchar(20) not null comment 'student name',
        stu_gender bit default 1,    -- add default value
        stu_birth date,
        tid int auto_increment unique,  -- auto increment no need specify when inserting # unique value
        school_id  int,
        primary key (stu_id),    -- primary key: uniquely identify a record  (ex. id)
        -- foreign key (school_id) references t_school(sch_id)    --many to one
    );
    alter table t_student add column stu_address varchar(255) first after school_id
        # first makes it first column, after make it after specific column
    alter table t_student drop column stu_address
    alter table t_student change/modify column stu_address stu_address varchar(511)

    alter table t_student add constraint fk_student_sch_id       --sch_id is primary key of school
        foreign key (school_id) references t_school(sch_id)
        # need update/drop t_student before update/drop t_school after foreign key created
        #or use cascade during foreign key creation, not recommended
        # foreign keys limitation: performance, not available for distributed clusters deployment
            # for large scale system use to build relation instead

    alter table t_student add constraint fk_student_sch_id foreign key(school_id) references
        t_school(sch_id) on delete cascade on update cascade
        # on delete set null (can't violate not null constraint)        restrict: default

    alter table t_student add constraint stu_id_sch_id unique(stu_id, sch_id);   # unique constraint
    alter table t_emp add constraint fk_emp_mgr foreign key (mgr) references t_emp(e_id)
    # can reference to attribute in own table (t_emp has e_id and mgr (manager))
    alter table t_emp add constraint ck_emp_dept check (dept between 0 and 100)
        # constraint for t_emp.dept [1,100], mysql doesn't support check


    insert into t_student values (101, 'Harry', 1, '1991-10-30', 'Hogwarts')
    insert into t_student (stu_id, stu_name) values (102, 'Harry Potter')
    insert into t_student (stu_id, stu_name,stu_gender) values (103,'Ronald',default),(104,'Hermione',0);

    truncate table t_student;
    delete from t_student where stu_id=102;

    update t_student set stu_address='Hogwarts', stu_birth='1991_09_30' where stu_id=103 or stu_id=104
        -- stu_id in (103,104)    -- stu_id between 101 and 105

    select * from t_student;
    select distinct stu_name, stu_address as address, cast(age as integer), age::integer, case when stu_gender=1 then
        'male' else 'female' end from t_student where stu_birth >= '1980-1-1' and stu_name like 'Har%' and stu_address
        is not null and grade in (1,2,3) and age not between 10 and 12 and
        cast(age as integer)  age::integer  cast data type to integer
        select age*2, age+year
        in (1,2,3)   # not in ('a','b','c')
        (grade, age) in (select max(grade), age from student group by age)
        select breed from cats group by breed order by count(breed) desc limit 1# find breed of cat with most population
        select  a.name, a.weight from cats a inner join (select avg(weight) mweight, breed from cats group by breed) b
            on a.breed=b.breed where a.weight>b.mweight      # find cat has higher weight than average in its breed
        ilike: ignore case,  between both side inclusive
        order by stu_birth desc, stu_id;
        # as alias    %: 0 or more any character   _: 1 character     <>: not equal
        is null    is not null   (can't use column=null)       order by asc default   desc
        CASE WHEN weight > 250 THEN 'over 250' WHEN weight > 200 AND weight <= 250 THEN '201-250'
            ELSE '175 or under' END AS weight_group        # case search function
        CASE sex WHEN '1' THEN 'male' WHEN '2' THEN 'female ELSE 'other' END   # case function
    select max(stu_birth) from t_student;         # (null is excluded)  min  max  sum(treat null as 0)
                                                  # avg(not include null)   count
    select NOW() - join_date::timestamp - interval '2 weeks' as period from t_student
        # ::timestamp   cast to time stamp      NOW() current date time    interval: time difference
    select stu_gender, count(stu_id) from t_student group by stu_gender having stu_id > 100;
        group by age, gender
        # use group by to limit aggregate function(count, max, sum,...) affected scope from whole table to part of table
        count(*) include null,   count(stu_id) doesn't include null    count(distinct age)
           round(age, 1) # round to specified decimal count    now()
        format(now(), 'YYYY-MM-DD')

    execution order: from & joins, where, group by, having, select, distinct, order by, limit & offset

    where -> group -> order
    select avg(score) as m from t_score group by stu_id having m>90
    use having to filter after group by (), can't use where because avg(score) is result after group by
    select stu_name from t_student where stu_birth = (select min(stu_birth) from t_student)   # sub query
    use where stu_id in (subquery)   if subquery have more than 1 result
        select * from (subquery) join (subquery) on id where id = (subquery)
    select LEFT(date, 10)  # substring start from left with length 10 characters
        RIGHT(date, LENGTH(date)-11)   # substring start from right with 11 character
        TRIM(both '()' FROM date)      # heading/trailing/both  remove front/end/both side of () from date
        LTRIM()  RTRIM()    # remove left/right side leading space
        length(name)        # return length of char
        POSITION('A' IN descript)     # get the index of first occurrence of 'A'
        SUBSTR(date, 4, 2) AS day      # substring of date   start position,length (start at 1), number of characters.
                                    #length optional, default length till end of string
        CONCAT(date, ', ', LEFT(date, 10)) as day   # concat column name or constant string
        UPPER(address), LOWER(address)   # convert to upper,lower case
        EXTRACT('year' FROM date), DATE_TRUNC('month', date)    # deconstruct and extract year of date data type object
                                                                # round dates to the nearest unit of measurement
        COALESCE(date, '1900/01/01')                            # replace null value with value

        CURRENT_DATE, CURRENT_TIME, CURRENT_TIMESTAMP, LOCALTIME, LOCALTIMESTAMP, NOW()

    select c1,c2,..., count/sum/avg/min/max(case when ... then ... when (a.id=b.id)... then ... else ... end) from table
         where ... group by ... (having ...)
    select a.c1, a.c2,..., b.d1, b.d2 from table1 a (inner/left/full outer) join (subquery) b on a.xx=b.xx / between
        b.xx and b.xx+3 where
    # use same columns value constraint to avoid Cartesian product (combination of two table items)
    select stu_name, sch_name from t_student t1, t_school t2 where t1.school_id=t2.sch_id
    select stu_name, avg_score from t_student t1, (select stu_id, avg(score) as avg_score from t_score
        group by stu_id) t2 where t1.stu_id=t2.stu_id
    select stu_name, ifnull(avg_score,0) from t_student t1 left join (select stu_id, avg(score) as
        avg_score from t_score group by stu_id) t2 on t1.stu_id=t2.stu_id limit 5
    select * from t_student stu join t_courses c on stu.course = c.id  # default inner join
        # join t_courses c where stu.course = c.id
    a inner join b on a.xid=b.xid inner join c on b.yid=c.yid where a.xid>10
    inner join: include data only match a.xid=b.xid constraint
    left/right/full outer join: include left table data even it don't match the on a.xid=b.xid constraint

    self join:  select distinct a.* from t_student a inner join t_student b on a.id=b.id where a.age>b.age
        # on a.id in (subquery)      # usually use to compare value in same table, subquery table need has alias
    select distinct a.num ConsecutiveNums from logs a, logs b, logs c where a.id+1=b.id and a.id+2=c.id and a.num=b.num
        and b.num=c.num       # select number of 3 consecutive id with same num

    ifnull(avg_score,0): return avg_score, if null return 0.
    limit 5: paging first 5 items       limit 5 offset 10  (or limit 10,5)  # skip 10 items get 5 items


    select * from a union select * from b     # append b to a with same number of columns with same data type and
                                              # remove duplicated row, not necessary same column name
        # union all     # keep duplicated value
    select * from a intersect select * from b   # get intersection of two query, unique result
    select * from a minus select * from b       # get items only in first query

    explain select stu_name from t_student where e_id =101   <>  not equal
    # show type (constant, all, range, searching type)  rows (how many lines to search)
        key (key related to search)

    # convert name course score to      name course name avg score
    SELECT name AS name_, MAX(CASE course WHEN "magic defense" THEN score ELSE 0 END) AS 'magic defense',
        MAX(CASE course WHEN "magic history" THEN score ELSE 0 END ) AS 'magic history', AVG(score) AS 'avg_score'
        FROM tb GROUP BY name


    select sum(score) over (partition by gender order by age) as total from t_student
        # partition similar function as group by but for window function, result scope will include previous cells
        #   value if there is order by, otherwise same value for each row in partition
        # partition by anf order by optional    select sum(score) over () from t_student
    select ROW_NUMBER() OVER () from t_student       # add row number column start at 1
        RANK() OVER (ORDER BY age)  # give same age value same rank number then skip count of same value   1,2,2,2,5
            # rank(), row_number() must have order by
        AVG(age) OVER (partition by house ORDER BY name )     # partition stack same attribute values rows together one
                # after another, not into one row(different from group by), first partition then order
        AVG(age) OVER (ORDER BY name ROWS between 1 preceding and 1 following)    # for current role get average from
                # 1 row above and 1 row below and current row
        DENSE_RANK()   # do not skip rank number after duplication    1,2,2,2,3
        NTILE(5) OVER (ORDER BY age)   # assign percentile value 1-5 based on age same order as age
        PERCENT_RANK()    CUME_DIST
        MIN(age)   MAX  SUM  COUNT  AVG
        sum(case when score>=60 then 1 else 0 end) as pass

        LAG(score, 1) OVER () as lag     # create lag column from score shift down(pull from previous row) one step
        LEAD()                         # pull from following row
        select duration -LAG(duration, 1) as diff OVER ()   # create column with duration difference from row above
        FIRST_VALUE()   LAST_VALUE()  # not in mysql
        nth_value(age, 3) over (order by time)   # return item age value from role with third smallest time value among
                                                # all rows above's time value

    select ROW_NUMBER() OVER win as rows from t_student where age < 10 WINDOW win as (ORDER BY age) ORDER BY age
        # use alias win and declare WINDOW, must declare it after where

    sdd EXPLAIN  in front of query to get rough execution time

    select * into new_table from old_table  # copy table
    select * from t_student where exists (query b)   # return first query result if query b has result. else no return

    # create index on frequent filtered and less modified value column
    INDEX idx_stu_name (stu_name),  during create
    create index idx_stu_name on t_student(stu_name);   # use extra space, constant search time
    drop index idx_stu_name on t_student;

    # create views  (snapshot of select query, create a temp table from query, query reuse (shorter code),
        limit visible column)
    create views vw_stu_sch as
    select stu_name, univ_name from t_student t1 inner join t_school t2 on t1.school_id=t2.school_id;
    select * from vw_stu_sch
    drop views vw_stu_sch

    # procedure   can't return value inside procedure, put return output inside input parameter
    # faster than use query, since answer is compiled and optimized ahead
    delimiter $$   # assign new delimiter switch from ; to $$
    create procedure p_stu_sch_avg_age(s_id int, out avg_age decimal(6,1))   -- return avg_age
    begin
        select avg(stu_birth) into avg_age from t_student where school_id=s_id;
    end$$
    delimiter ;
    call p_stu_sch_avg_age(20, @a);     select @a from dual;
    drop procedure p_stu_sch_avg_age;

    # function
    delimiter $$
    create function genPerson(@name varchar(20)) returns varchar(50)   -- return avgage
    begin
        declare @cmd varchar(50) default '';   # declare variable
        set @cmd=concat('create table ',@name,' (id int, name varchar(20));');
        while @cmd <= 5:
            PRINT @cmd
            if @cmd <= 5:
                then PRINT @cmd
            elseif @cmd <= 5:
                then PRINT @cmd
            else PRINT @cmd
            end if
        return @cmd;
    end$$
    delimiter ;
    select genPerson('student')
    drop function if exists genPerson

    # trigger
    trigger is not used to. when do some operation, some other operation defined by trigger is done in
        background automatically. but in reality might cause lock the table for parallel operation,
        and decrease performance

    create user 'harry'@'%' identified by '123456'  # create user login on any location with password
        '%': any location  '120.10.20.30':specific ip   'localhost': this computer
    grant all privileges on university.*  to 'harry'@'%'   # grant harry all access to university database
    revoke insert, delete, update on university.*  from 'harry'@'%' # remove insert, delete, update access
    drop user 'harry'@'%'  # remove user

    # transaction  change multiple actions into atomic
    begin;   # start transaction
    operation 1; operation 2;
    commit;  # make all operation in effect
    rollback;  # undo all operation

    # one to one relationship
    alter table t_id_card add constraint fk_card_pid foreign key(pid) reference t_person(p_id);
    alter table t_id_card add constraint uni_card_pid unique (pid)
        #pid varchar(20) unique,  during creation

    Important notes
    # recommend use lower case for table and database name
    # data search result case-sensitive or not depends on collate rules during database creation
        # utf8_general_ci not case-sensitive,  utf8_bin  is case-sensitive
    # database object name better use prefix to distinguish: table, views, index, function, procedure, trigger
    # not recommend use in, not in, distinct. consider use exists, no exists
    select 'x'  #'x'    select 'x' from dual  # 'x'  table dual is unique non exist table
    select 'x' from t_student  # how many row is how many 'x'
    select 'x' from dual where exists (select * from t_student where stu_name like 'Harry%')
        # 'x's if some student name harryxxx
    select emp_name, title from t_emp t1 where exists (select 'x' from t_emp t2 where t1.emp_id=t2.mgr)
    == select distinct emp_name, title from t_emp t1 inner join t_emp t2 where t1.emp_id=t2.mgr
    # in subquery use smaller table compare to the main query
    # avoid use <>  not equal, slow
    # use index on frequent searched item, index help on =, like 'keyword%',
        but not helpful for <>, like '%keyword'
    # index slower insert update delete, faster search. use extra space

    conn = pymysql.connect(host='localhost', port=3306, user='cai', password='123456',
                           database='company', charset='utf8')
    with conn.cursor() as cursor:
        cursor.execute('SELECT * from table where id = %(some_id)d', {'some_id': 1234})   # return item count
        res = cursor.fetchall()    # return actual data
        result = cursor.execute("INSERT INTO t_emp VALUES(%s,%s,%s,%s,%s, %s,%s,%s)",
                    (int(e['id']), e['name'], int(e['gender']), int(e['mgr']), float(e['salary']),
                     e['address'],e['birth'],int(e['dept'])))


    large scale distributed database
        performance drop after 5 millions record in a table
        data sharding: split User table into user_0, user_1, user_2,... to increase operations performance
        horizontal slicing
            retrieve item in specific table searching via (id // count per table)
                but each database has concurrency limit

            vertical slicing
                split original table via columns, group columns into different tables base on functionality, link via id

        split into multiple databases
            mainly use horizontal slicing, save id range of records inside multiple databases

        duplication
            master, multiple slaves have same data. slave update data once master changes
            separate read write: write data into master node, read from slaves node. (use config file)
            zookeeper: manage distributed system. if master died will poll for new master among slaves.
                need reliable crossover point(hard for duplication): such as DNS(domain name resolution), load balancer


    # Redis
    key-value database, save in memory. single thread+ async I/O (Coroutines)
    high speed cache  (move frequently used data from database to memory)
    message queue

    redis.io   redisdoc.com
    wget http://download.redis.io/releases/redis-5.0.4.tar
    gunzip redis-5.0.4.tar.gz    tar -xvf redis-5.0.4.tar    cd redis-5.0.4/
    make && make install
    ~     redis-server --version    redis-cli --version        redis-sentinel --version
    or yum install redis  # older version

    redis-server      # start redis server at backend tcp port 6379    --port 1234
    redis-server --requirepass 123456     # add password     --appendonly yes  change way of saving (aof), default rdb
        --bind xxx.xxx.xx.xx > redis.log     get ip from ifconfig eth0     save console log to redis.log
    jobs  # show running process    fg%1  move to foreground   Ctrl+z  stop process  bg%1 start in background

    redis-server redis-5.0.4/redis.conf    # use config file
    redis-client  auth 123456      shutdown nosave
    redis-cli  # connect to redis client   -h 127.33.1.23  -p 1234   default 127.0.0.1:6379

    close redis server options: 1. server  ctrl+c    2. kill process # (ps -ef | grep redis)
        3. redis-cli   shutdown  quit

    #windows
        https://github.com/microsoftarchive/redis/releases    download install Redis-x64-3.0.504.msi
        open redis-server   redis-cli

    ping                  # return pong if connected

    String key-value
    set username harry    # set key value
    setnx username ronald  # set key value only if key doesn't exist, else do nothing
    set 12345 100 ex 30  # set key (12345) value (100) which expire in 30 sec
                         # equivalent to setex 12345 30  100
    ttl 12345            # (int) 25    check time to live
    mset k1 v1 k2 v2     # add multiple key value key value
    append username potter  # harrypotter   append values to key
    get username         # "harrypotter"    (nil): empty value
    getrange username 2 5  # return substring of value  index start 0,   [2,5] rryp
    mget username k1     # get multiple values from keys
    incr score           # add one to value of key score
    incrby score 10      # add 10 to value of key score
    exists username      # return 1 if exist key username
    del username         # delete key username
    keys user*           # search key  * match any number character  ? match one character
                         # keys *   show all keys
    hashtable
    hset dictname dictkey "dictvalue"   # HSET [hash] [field] [value]
                         # return 1 if added, if already exist return 0
    hmset dictname k1 v1 k2 v2   # HMSET [hash] [field] [value] [field] [value]
    hget dictname k1     # get dictname key k1 value
    hmget dictname k1 k2     # get dictname key k1 value
    hkeys dictname       # get dictname all keys
    hvals dictname       # get dictname all values
    hgetall dictname     # get dictname all keys value
    hdel dictname k1     # delete k1 v1 of dictname

    list
    lpush number 10 20 30  # on the left side adding 10 20 30
    lrange number 0 -1   # "30" "20" "10"  check number index 0 to last
    lpop number          # "30"  left side remove one item and return
    lindex number 0      # 1  return the item at index 0, nil if index out of range
    llen number          # 3  return the length of list number

    set
    sadd set1 10 20 10 30  # add item to set
    srem set1 10 20      # remove 10, 20 from set1
    smembers set1        # "10" "20" "30"  return all set items
    scard set1           # 3  return count of items
    sismember set1 10    # 1  true  0 false  check item is inside set
    sinter set1 set2     # return intersection of sets
    sunion set1 set2     # return union of sets
    sdiff set1 set2      # return item only in set1
    spop set1 2          # randomly delete 2 items in set1 and return

    zset                 # ordered set
    zadd scoreboard 10 a 50 b 30 c  # add items and relative score to z set scoreboard
    zrange scoreboard 0 -1     # "a" "c" "b"  return item with score small to large
    zreverange scoreboard 0 -1  # "b" "c" "a"  return item with score large to small
    zincrby scoreboard 50 a    # add 50 score to a in scoreboard

    geography   (build with zset)
    geoadd map 100.1 30.2 loc1 100.9 31.2 loc2   # add locations longitude latitude to map
    geodist map loc1 loc2 km   # return line distance between loc1 and loc2 in km
    georadius map 100 31 10 km  # "loc1" "loc2"  return locations within 10km radius of given longitude latitude
        georadius map 100 31 10 km withdist   # result also show distance between given and answer location

    save / bgsave        # save immediately    save at background immediately

    type username        # string   return type of key
                         # redis data type: string, list, array, hash, set, zset (ordered set)
    select 1             # change to database 1, default 0. total default 16 databases
    randomkey            # randomly return a key in the database
    dbsize               # return total keys in current database
    flushdb              # empty database, delete all data in current database
    flushall             # delete all data in all databases

    quit /shutdown        # end redis-cli

    redis multi machine read write (master slave)
    replica>  replicaof 120.33.22.13 6379   # assign this computer to the slave of ip 120.33.22.13 port 6379
    replica> masterauth 123456    # type in the auth of master
    or redis-server --replicaof 120.33.22.13 6379  --masterauth 123456 > redis.log &
        # replica node can't write but can read master's data
    slave> replicaof no one   # terminate slave state

    master> info replication   # check whether have slaves

    redis-benchmark  # test speed   40000+ /sec
    redis-check-rdb --fix dumb.rdb  # check and repair dump.rdb for cached data which causing issue after crash
    redis-check-aof --fix appendonlu.aof   # recover crash aof saving mode

    connection problem: add ip to whitelist in server, check firewall

    import redis
    conn = redis.Redis(host='127.0.0.1', port=6379, password=123456, db=1)
    conn.flushdb()
    conn.set('user1', 'Harry', ex=300)


    # mongodb
    MongoDB save data as documents. data structure: key -> value, similar to json. value can be other document, array,
        document array. Able to set index for any attribute. able to create mirror for extension from local or internet.
        Use update() to replace document or update attribute. Use map/reduce for batch operation. support many languages.
    plugins and tools: Munin(network, system monitoring), Gangila(system monitoring), Cacti(gui network, cpu monitoring)
        many GUI tools (compass, Studio 3T)
    MongoDB stores data in BSON format (UTF-8 encoded).


    install mongodb, mongodb shell, mongo compass
    net start/stop MongoDB    # start/stop mongodb service
    mongosh    # open mongo shell to interact with mongodb
    terminology: collection (SQL table),  document (SQL row),  field (SQL column)
    mongodb shell language use javascript

    database name can't use: space, empty string,  .  $  /  \  \0, must lowercase, max 64 bytes, admin, local, config
    key name can't use: space \0  ($  _ careful use)
    collection name can't use: empty string  \0,   start with system. ($ careful use)
    data no need same schema or datatypes, case sensitive, documents (key-value pair) are ordered, no duplicate keys
        (field) inside one document, keys are string

    mongodb doesn't have transaction, but document save, update, delete are all atomic operation.

    capped collection:  fix size (max 1e9 bytes), high performance collection, data order same as insertion order, when
        update, don't exceed collection size to keep in same location. can't delete one document, need drop() all
        documents in the collection
        db.createCollection("cap_coll", {capped:true, size:100000,max:1000})  # size 1e5 bytes, max 1000 documents
        db.cap_coll.isCapped()  # return whether capped collection
        db.runCommand({"convertToCapped":"Hogwarts",size:10000})   # convert collection to capped


    datatype:
        String (utf-8), Integer, Boolean, Double, Array, Timestamp, Object, Null, Symbol(similar to string, for special
            characters), Date, ObjectID(similar to primary key, include timestamp, machine code, process id, random
            number), Binary Data, Code(javascript code), Regular expression, Min/Max keys
            each document has a "_id" field, default ObjectID type, unique identifier. not autoincrement because not
                efficient .
            var newObject = ObjectId()
            newObject.getTimestamp()    newObject.str

    # connect to mongodb
    uri: mongodb://[username:password@]host1[:port1][,host2[:port2],...[,hostN[:portN]]][/[database][?options]]
        port default 27017, at least one host, default database: test, options:/?replicaSet=name;slaveOk=true...


    # create / show / switch database
    show dbs        # show all database
    db              # show current database
    use hogwarts       # switch to hogwarts database, create if not exist

    # create collection
    db.createCollection("hogwarts_table", {capped: true, size: 6142800, max: 10000 })
        # collection name, options  capped collection, fix size in bytes, max documents

    # show collection
    show collections

    # drop database / collection
    db.dropDatabase()  # remove current database
    db.hogwarts.drop()    # drop collection hogwarts

    # insert document
        automatically generate _id for document
    db.hogwarts_table.insert({name: 'Harry Potter',age:10})
    db.hogwarts_table.insertOne({name: 'Harry Potter'})
    db.hogwarts_table.insertMany([{name: 'Harry Potter'}, {name: 'Hermione Granger'}])

    # update document
    db.collection.update(<query>,<update>,{upsert: <boolean>,multi: <boolean>,writeConcern: <document>}
        <query>: where condition, update: $inc  $set, upsert:default false, not insert if not exist, multi: default
        # following operations are atomic operations
        {$set: {field: value}}  # update value for field, will not create if not exist
        {$unset: {field: 1}}  # remove a field
        {$inc: {field: value}}  # increase value for field
        {$push: {field: value}}  # append value to a array field, create array if field not exist
        {$pushAll: { field: value_array}}  # append multiple values
        {$pull: {field: value}    # remove value from field
        {$addToSet: {field: value}  # append value if not exist
        {$pop: {field: 1}}   # remove first or last value in array field
        {$rename: {old_field_name: new_field_name}}   # rename field
        {$bit: {field: {and : 5}}}   # bitwise operation


        false, only update first match, writeConcern: throw exception level
    db.hogwarts_table.update({'name':'Harry Potter'},{$set:{'name':'Harry'}}, {multi:true})
    db.hogwarts_table.update({'name':'Harry Potter'},{$unset:{'name':1}})   # remove key: name

    # delete document
    db.collection.remove(<query>,{justOne: <boolean>,writeConcern: <document>})
        <query>: where condition,  justOne: default false, remove all matched. if true/1 only delete one document
        writeConcern: throw exception level
    db.hogwarts_table.remove({"name":'Harry'})
    db.hogwarts_table.remove({})  # remove all data

    # find document
    db.collection.find(query, projection)
        <query>: where condition, projection: specify column to display in result, 0 not display, 1 displayed
        query operations: age:{$lt:50}   lte   gte  gt  ne  $or:[{k:v},{k:v}]  $in:[v1,v2]   nin (not in)   $type
        .limit(NUMBER)  .skip(NUMBER)  .sort({KEY:1}) (1:asc, -1:desc)
    db.hogwarts_table.find({},{"name":1})   # {} query is empty
    db.hogwarts_table.find({"name": 'Harry', $or: [{"age":{$lt:50}},{"age":{$gt:80}}]})
    db.hogwarts_table.find({"name":null).limit(2).skip(1) # find null value, skip first result. only get 2 documents,
    db.hogwarts_table.find({"name":/^Har/})   # sql like
    db.hogwarts_table.find({"name":'Harry', 'age':10})
    db.hogwarts_table.find({"name": {$type:'string'}})  # find by dtype
    db.hogwarts_table.distinct('name')   # find distinct name
    db.hogwarts_table.find().sort({'name':-1})  # sort by name desc
    db.hogwarts_table.findOne({"name":'Harry', 'age':10})   # return one data

    # index
    mongodb create index in RAM, if overflow, it will remove some indexes. can't search by $in, $nin, $mod, $where
    collection max 64 indexes, index name max 128 bytes, composite index max 31 fields.

    db.collection.createIndex(keys, options)
        {k: 1} # asc,   -1 desc
        options: background, unique, name, dropDups, sparse, expireAfterSeconds, v, weights, default_language,
            language_override
    db.hogwarts_table.createIndex({'name': 1, 'age': 1}, {background: true})
    db.hogwarts_table.createIndex({'name': 1},{"name":'idx_name'})  # add index name
    db.hogwarts_table.getIndexes()   # get collection index
    db.hogwarts_table.totalIndexSize()
    db.hogwarts_table.dropIndexes()
    db.hogwarts_table.dropIndexes('index_name')
    db.hogwarts_table.createIndex({'course.name':1})  # create sub index for document field


    db.hogwarts_table.find({name:'Harry',_id:0}).hint({name:1})  # force use index {name:1}
    db.hogwarts_table.find({name:'Harry',"age":{$lt:50}}).explain()  # to check whether use index

    # text search
    db.posts.createIndex({intro:"text"})   # intro text field
    db.hogwarts_table.find({$text:{$search:"my name"}})  # search my name in the intro field
        db.hogwarts_table.find({$text:{$regex:"my name",$options:"$i"}})   #search via regular expression, ignore case
            or db.hogwarts_table.find({post_text:/my name/})

    # best performance if searching by same fields(or extra _id) as combined index fields


    # aggregation
    db.COLLECTION_NAME.aggregate(AGGREGATE_OPERATION)
    sum, avg, min, max, push (add data), addToSet, first, last
    $project: modify document schema, rename, add, delete field
    $match: filter data, match only satisfied (having)
    $limit: limit the return document count
    $skip:  skip first # of document in matched result
    $unwind: split array type field in document into multiple document, one for each document
    $group: groupby operation for analyze
    $sort: return sorted result

    db.hogwarts_table.aggregate([{$group : {'_id': "$name", 'score_avg' : {$avg : '$score'}}}])   # group by
        # same as select name, avg(score) as score_avg from hogwarts_table group by name
        # name and take score average    $group: {'_id': null, 'total':{$sum:'$score'}}   # null means aggregate all
    db.hogwarts_table.aggregate([{$group: {'_id': "$name", 'friend': {$push: 'Ronald'}}}])
    db.hogwarts_table.aggregate([{{$group: {_id: null, count: {$sum: 1}}},{$match: {'score': { $gt: 70, $lte: 90}}}])
        # return count of documents with 70<score<90, $sum:1, 1 means add 1 to sum for each match


    # date
    var date1 = new Date()   or  var date1 = Date()   # ISODate("2018-03-04T14:58:51.233Z")  current date
    typeof date1  # Object    typeof date1.toString()  # String


    # data redundancy
    one primary node, one or multiple secondary node. master take client read / write request,
        secondary nodes copy primary node operation logs periodically and make same operations to keep sync
    mongod --port "PORT" --dbpath "YOUR_DB_DATA_PATH" --replSet "REPLICA_SET_INSTANCE_NAME"
    mongod --port 27017 --dbpath "D:\mongodb\data" --replSet rs0

    client side:
    rs.initiate()    # initialize new replica set
        rs.initiate({_id: 'rs0', members: [{_id: 0, host: 'localhost:27020'}, {_id: 1, host: 'localhost:27021'}]})
    rs.conf()   # check replica set config
    rs.status()  # check replica set status
    rs.add(HOST_NAME:PORT)   # add replica to replica set, host name is the primary mongodb node

    for large dataset, use distributed database
    several query routers take client requests, make cluster look like one database. several Config servers (mongod
        instances) store the clusterMetadata. Several Shards store data, can have replica set to prevent single point
        failure.
    #without replica
    /usr/local/mongoDB/bin/mongod --port 27020 --dbpath=mongoDB/shard/s0 --logpath=mongoDB/shard/log/s0.log --logappend
        --fork      # start multiple shard server
    # with replica (1 server 2 replica)
    nohup mongod --port 27020 --dbpath=/data/db1 --logpath=/data/log/rs0-1.log --logappend --fork --shardsvr --replSet=rs0 
    nohup mongod --port 27021 --dbpath=/data/db2 --logpath=/data/log/rs0-2.log --logappend --fork --shardsvr --replSet=rs0
        # nohup: nonstop running
    rs.initiate({_id: 'rs0', members: [{_id: 0, host: 'localhost:27020'}, {_id: 1, host: 'localhost:27021'}]})
        # duplicate this with another shard server

    /usr/local/mongoDB/bin/mongod --port 27100 --dbpath=/mongoDB/shard/config --logpath=/mongoDB/shard/log/config.log
        --logappend --fork     # start config server

    nohup mongod --port 27100 --dbpath=/data/conf1 --logpath=/data/log/conf-1.log --logappend --fork --configsvr
        --replSet=conf    # config server with replica, duplicate this with another port and locations
    rs.initiate({_id: 'conf', members: [{_id: 0, host: 'localhost:27100'}, {_id: 1, host: 'localhost:27101'}]})

    /usr/local/mongoDB/bin/mongos --port 40000 --configdb localhost:27100 --fork --logpath=/mongoDB/shard/log/route.log
        --chunkSize 500      # start query router, default 200 mb
    nohup mongos --port 40000 --configdb conf/localhost:27100,localhost:27101 --fork --logpath=/data/log/route.log
        --logappend     # with replica start query router

    # in mongo shell
    /usr/local/mongoDB/bin/mongo admin --port 40000
    db.runCommand({ addshard:"localhost:27020" })  # add shard server
        # db.runCommand({ addshard: 'rs0/localhost:27020,localhost:27021'})   # with replica
    db.runCommand({ enablesharding:"test" }) # set which database allow sharding
    db.runCommand({ shardcollection: 'test.user', key: {name: 1}})  # set which collection to use sharding


    # backup and recover data
    mongodump -h host -d dbname -o copy_dir    # backup: host port of mongodb, database name to copy, and copy directory
        --host  --port  --out  --db    default localhost, 27017 /dump
    mongorestore -h <hostname><:port> -d dbname <path>
        --host  --db  --dir  --drop (drop backup then copy)

    # monitoring
    terminal:  mongostat    # running status, performance
    terminal:  mongotop     #track mongodb instance read write time


    # relationship
    # manual reference
    either embed the whole many objects to one object or embed their references
    {   # embed objects relationship
       "_id":ObjectId("52ffc33cd85242f436000001"),
       "name": "Tom",
       "course": [
          {"name": "Magic Defence"},
          {"name": "Magic History"}
       ]
    }
    db.hogwarts_table.findOne({"name":"Tom"},{"course":1})   # get address

    {   #  references relationship
       "_id":ObjectId("52ffc33cd85242f436000001"),
       "name": "Tom Benzamin",
       "course_ids": [
          ObjectId("52ffc4a5d85242602e000000"),
          ObjectId("52ffc4a5d85242602e000001")
       ]
    }
    var result = db.hogwarts_table.findOne({"name":"Tom"},{"course_ids":1})
    db.course_table.find({"_id":{"$in":result["course_ids"]}})

    # DBRef reference  (across multiple db, collections)
    # document reference
    inside one document can reference other document data
    { $ref: , $id: , $db:  }   # collection name, object id, database name
    {
       "_id":ObjectId("53402597d852426020000002"),
       "name": "Tom",
       "course": {
       "$ref": "course_table",
       "$id": ObjectId("534009e4d852427820000002"),
       "$db": "Hogwarts"},
    }
    var dbRef = db.hogwarts_table.findOne({"name":"Tom"}).course
    db[dbRef.$ref].findOne({"_id":(dbRef.$id)})    # find course with collection name, id
        # {"_id":ObjectId(dbRef.$id)}


    # mapreduce
    db.collection.mapReduce(function() {emit(key,value);}, function(key,values) {return reduceFunction},
        {out: collection,query: document,sort: document,limit: number})
    db.posts.mapReduce(function() {emit(this.name, this.score);}, function(key, values){return Array.sum(values)},
        {query:{"age":{$lt:50}}, out:"score_total" }).find()
        # map with name as key and put all score into an array, then reduce by sum, query filter out results, and
        add a collection 'score_total'.  { "_id" : "Harry", "value" : 4 }


    # GridFS
    GridFS used to store files greater than 16 MB (images, audio, video). GridFS is a way to store in collections via
        sharding file into chunks (default 256 KB each), each chunk is store inside a collection as one document. GridFS
        use 2 collection to store a file (fs.files (meta data: file name, size, type...), and fs.chunks (binary data))
        mongofiles.exe -d gridfs put song.mp3
        db.fs.files.find()   # find fs.files information
        db.fs.chunks.find({files_id:ObjectId('534a811bf8b4aa4d33fdf94d')})   # find all documents(chunks)

    client = pymongo.MongoClient("mongodb://localhost:27017/")
    res = client.hogwarts.hogwarts_table.create_index({'name': 1}, unique=True)
"""