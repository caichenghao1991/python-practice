"""
    database: data persistence, retrieval and management
    relation database(SQL): need accuracy Oracle/ MySQL / MariaDB (similar to MySQL)   (row-column, 2d table)
        one line: a record    one column: attribute
    non-relational database (NoSQL): faster flexible. MongoDB, Redis (key-value pair, cache),
        Elasticsearch (search engine)
    primary key: uniquely identify a record  (ex. id)
    SQL: structured query language
        DDL data definition language:  create  drop  alter
        DML data manipulation language:  select insert delete update    crud
        DCL data control language:   grant  revoke

    rpm -ivh mysql-5.7.25-1.el7.x86_64.rpm    # install MySQL
    rpm -e mysql-5.7.25-1  # uninstall mysql

    yum install -y mariadb mariadb-server  # install MariaDB
    systemctl start/stop/status mariadb  # start maria server on centos 8
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
    create database university default charset utf8;     # utf8mb4  emoji
    drop database if exists university;   drop database university;
    use university;        # switch to university database context
    show tables;       # show tables in the database

    drop table if exists t_student;
    create table t_student   -- create table student
    (   stu_id int not null,
        stu_name varchar(20) not null comment 'student name',
        stu_gender bit default 1,    -- add default value
        stu_birth date,
        tid int auto_increment,  -- auto increment no need specify when inserting
        school_id  int,
        primary key (stu_id),
        -- foreign key (school_id) references t_school(sch_id)    --many to one
    );
    alter table t_student add column stu_address varchar(255)
    alter table t_student drop column stu_address stu_address varchar(511)
    alter table t_student change column stu_address
    alter table t_student add constraint fk_student_sch_id       --sch_id is primary key of school
        foreign key(school_id) references t_school(sch_id)
    alter table t_student add constraint stu_id_sch_id unique(stu_id, sch_id);   # unique constraint

    insert into t_student values (101, 'Harry', 1, '1991-10-30', 'Hogwarts')
    insert into t_student (stu_id, stu_name) values (102, 'Harry Potter')
    insert into t_student (stu_id, stu_name,stu_gender) values (103,'Ronald',default),(104,'Hermione',0);

    truncate table t_student;
    delete from t_student where stu_id=102;

    update t_student set stu_address='Hogwarts', stu_birth='1991_09_30' where stu_id=103 or stu_id=104
        -- stu_id in (103,104)    -- stu_id between 101 and 105

    select * from t_student;
    select distinct stu_name, stu_address as address, case stu_gender when 1 then 'male' else 'female' end
        from t_student where stu_birth >= '1980-1-1' and stu_name like 'Har%' and stu_address is not null
        order by stu_birth desc, stu_id;
        # as alias    %: 0 or more any character   _: 1 character     <>: not equal
        is null    is not null          order by asc default   desc
    select max(stu_birth) from t_student;         # (null is excluded)  min  max  sum  avg   count
    select stu_gender, count(stu_id) from t_student group by stu_gender having stu_id > 100;
    where -> group -> order
    select avg(score) as m from t_score group by stu_id having m>90
    use having to filter after group by (), can't use where because avg(score) is result after group by
    select stu_name from t_student where stu_birth = (select min(stu_birth) from t_student)   # sub query
    use where stu_id in (subquery)   if subquery have more than 1 result



    redis.io   redisdoc.commit
    wget http://download.redis.io/releases/redis-5.0.4.tar
    gunzip redis-5.0.4.tar.gz    tar -xvf redis-5.0.4.tar    cd redis-5.0.4/
    make && make install
    ~     redis-server --version    redis-cli --version        redis-sentinel --version
    redis-server &     # start redis server at backend tcp port 6379
    redis-server --requirepass 123456 &    #add password
    redis-client  auth password 123456      shutdown nosave
    close redis server options: 1. server  ctrl+c    2. kill process # (ps -ef | grep redis)
        3. redis-cli   shutdown  quit


"""