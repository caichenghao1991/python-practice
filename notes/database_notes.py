"""
    database: data persistence, retrieval and management
    relation database(SQL): need accuracy Oracle/ MySQL / MariaDB (similar to MySQL)   (row-column, 2d table)
        one line: a record    one column: attribute
        ACID
        Atomic: All operations in a transaction succeed or every operation is rolled back.
        Consistent: On the completion of a transaction, the database is structurally sound.
        Isolated: Transactions do not contend with one another. Contentious access to data is moderated
            by the database so that transactions appear to run sequentially.
        Durable: The results of applying a transaction are permanent, even in the presence of failures.

    non-relational database (NoSQL): faster flexible. MongoDB, Redis (key-value pair, cache),
        Elasticsearch (search engine)
        BASE
        Basic Availability: The database appears to work most of the time.
        Soft-state: Stores don’t have to be write-consistent, nor do different replicas have to be
            mutually consistent all the time.
        Eventual consistency: Stores exhibit consistency at some later point (e.g., lazily at read time).

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
        tid int auto_increment,  -- auto increment no need specify when inserting
        school_id  int,
        primary key (stu_id),    -- primary key: uniquely identify a record  (ex. id)
        -- foreign key (school_id) references t_school(sch_id)    --many to one
    );
    alter table t_student add column stu_address varchar(255) first after school_id
        # first makes it first column, after make it after specific column
    alter table t_student drop column stu_address
    alter table t_student change/modify column stu_address stu_address varchar(511)

    alter table t_student add constraint fk_student_sch_id       --sch_id is primary key of school
        foreign key(school_id) references t_school(sch_id)
        # need update/drop t_student before update/drop t_school after foreign key created
        #or use cascade during foreign key creation, not recommended
    alter table t_student add constraint fk_student_sch_id foreign key(school_id) references
        t_school(sch_id) on delete cascade on update cascade
        # on delete set null (not violating not null column)        restrict: default

    alter table t_student add constraint stu_id_sch_id unique(stu_id, sch_id);   # unique constraint
    alter table t_emp add constraint fk_emp_mgr foreign key (mgr) references t_emp(e_id)
    # can reference to attribute in own table (t_emp has e_id and mgr (manager))

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

    # use same columns constraint to avoid Cartesian product (combination of two table items)
    select stu_name, sch_name from t_student t1, t_school t2 where t1.school_id=t2.sch_id
    select stu_name, avg_score from t_student t1, (select stu_id, avg(score) as avg_score from t_score
        group by stu_id) t2 where t1.stu_id=t2.stu_id
    select stu_name, ifnull(avg_score,0) from t_student t1 left outer join (select stu_id, avg(score) as
        avg_score from t_score group by stu_id) t2 on t1.stu_id=t2.stu_id limit 5
    a inner join b on a.xid=b.xid inner join c on b.yid=c.yid
    inner join: include data only match a.xid=b.xid constraint
    left/right/full outer join: include left table data even it don't match the on a.xid=b.xid constraint
    ifnull(avg_score,0): return avg_score, if null return 0.
    limit 5: paging first 5 items       limit 5 offset 10  (or limit 10,5)  # skip 10 items get 5 items

    explain select stu_name from t_student where e_id =101   <>  not equal
    # show type (constant, all, range, searching type)  rows (how many lines to search)
        key (key related to search)

    # create index on frequent filtered and less modified value column
    INDEX idx_stu_name (stu_name),  during create
    create index idx_stu_name on t_student(stu_name);   # use extra space, constant search time
    drop index idx_stu_name on t_student;

    # create view  (snapshot of select query, create a temp table from query, query reuse (shorter code),
        limit visible column)
    create view vw_stu_sch as
    select stu_name, univ_name from t_student t1 inner join t_school t2 on t1.school_id=t2.school_id;
    select * from vw_stu_sch
    drop view vw_stu_sch

    # procedure   can't return value inside procedure, put return output inside input parameter
    # faster than use query, since answer is compiled andoptimized ahead
    delimiter $$   # assign new delimiter switch from ; to $$
    create procedure p_stu_sch_avg_age(s_id int, out avgage decimal(6,1))   -- return avgage
    begin
        select avg(stu_birth) into avgage from t_student where school_id=s_id;
    end$$
    delimiter ;
    call p_stu_sch_avg_age(20, @a);     select @a from dual;
    drop procedure p_stu_sch_avg_age;

    # function
    delimiter $$
    create function genPerson(name varchar(20)) returns varchar(50)   -- return avgage
    begin
        declare cmd varchar(50) default '';   # declare variable
        set @tableName=name;
        set cmd=concat('create table ',@tableName,' (id int, name varchar(20));');
        return str;
    end$$
    delimiter ;
    select genPerson('student')

    # trigger
    trigger is not used to. when do some operation, some other operation defined by trigger is done in
        background automatically. but in reality might causing lock the table for parallel operation,
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
    start transaction


    Important notes
    # recommend use lower case for table and database name
    # data search result case sensitive or not depends on collate rules during database creation
        # utf8_general_ci not case sensitive,  utf8_bin  is case sensitive
    # database object name better use prefix to distinguish: table, view, index, function, procedure, trigger
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