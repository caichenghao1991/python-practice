import pymysql


class Department:
    def __init__(self, d_id, name, address):
        self.id = d_id
        self.name = name
        self.address = address

    def __str__(self):
        return f'{self.id}\t{self.name}\t{self.address}'


def get_connection():
    conn = pymysql.connect(host='localhost', port=3306, user='cai', password='123456',
                           database='company', charset='utf8')  # default cursor return tuple,
    # add cursorclass=pymysql.cursors.DictCursor to change to list of dictionary
    return conn


def insert(emp_list, conn):
    try:
        with conn.cursor() as cursor:
            for e in emp_list:
                result = cursor.execute(
                    "INSERT INTO t_emp VALUES(%s,%s,%s,%s,%s, %s,%s,%s)",
                    (int(e['id']), e['name'], int(e['gender']), int(e['mgr']), float(e['salary']),
                     e['address'],
                     e['birth'],
                     int(e['dept']))
                )

            if result == 1:
                print("insert succeeded")
                conn.commit()
    except pymysql.MySQLError as error:
        print(error)
        conn.rollback()


def select(conn):  # cursor default return tuple
    try:
        with conn.cursor() as cursor:
            cursor.execute('select d_id, d_name, d_address from t_dept')
            for r in cursor.fetchall():  # return one tuple of tuples  fetchmany(int)  fetchone return a tuple
                print(f'department id: {r[0]}')
                print(f'department name: {r[1]}')
                print(f'department address: {r[2]}')
                print('-' * 20)
    except pymysql.MySQLError as error:
        print(error)


def select2():  # cursor return dictionary with cursorclass
    conn = pymysql.connect(host='localhost', port=3306, user='cai', password='123456',
                           database='company', charset='utf8', cursorclass=pymysql.cursors.DictCursor)
    # default cursor return tuple, add cursorclass=pymysql.cursor.DictCursor to change to list of dictionary
    try:
        with conn.cursor() as cursor:
            cursor.execute('select d_id, d_name as name, d_address as address from t_dept')
            for r in cursor.fetchall():  # return one tuple of tuples  fetchmany(int)  fetchone return a tuple
                print(f'department id: {r["d_id"]}')  # default dictionary key is column name
                print(f'department name: {r["name"]}')  # use alias for dictionary key
                print(f'department address: {r["address"]}')
                dept = Department(**r)
                print(dept)
                print('*' * 20)
    except pymysql.MySQLError as error:
        print(error)
    finally:
        conn.close()


def insert0(conn):
    try:
        with conn.cursor() as cursor:
            result = cursor.execute(
                'insert into t_dept values (1, "Information Technology", "Hogwarts");')
            if result == 1:
                print("insert succeeded")
                conn.commit()
    except pymysql.MySQLError as error:
        print(error)
        conn.rollback()


def insert2(conn):
    try:
        with conn.cursor() as cursor:
            result = cursor.execute(
                'insert into t_dept values ("{}", "Magic", "Hogwarts")'.format(2))  # don't use string format
            # string concatenation can cause sql injection
            # select * from t_user where username='{}' and password='{}'   admin  x' or '1'='1'; update ... where '1'='1
            # select * from t_user where username='admin' and password='x' or '1'='1'; update ... where '1'='1
            if result == 1:
                print("insert succeeded")
                conn.commit()
    except pymysql.MySQLError as error:
        print(error)
        conn.rollback()
    ''' 
         CREATE DATABASE company;
         USE company;
         DROP TABLE IF EXISTS t_dept;
         CREATE TABLE t_dept(
            d_id INT NOT null,
            d_name VARCHAR(50) NOT NULL,
            d_address VARCHAR(100),
            INDEX i_d_name (d_name),
            PRIMARY KEY (d_id)
        );
         DROP TABLE IF EXISTS t_emp;
         CREATE TABLE t_emp(
             e_id INT AUTO_INCREMENT NOT null,
             e_name VARCHAR(50) NOT NULL,
             e_gender bit DEFAULT 1, # 1 is male, 0 is female
             e_mgr INT,   # manager id,
             e_salary FLOAT,
             e_address VARCHAR(100),
             e_birth DATE,
             e_dept INT,
             INDEX i_e_name (e_name),
             PRIMARY KEY (e_id),
             FOREIGN KEY (e_dept) REFERENCES t_dept(d_id),
             FOREIGN KEY (e_mgr) REFERENCES t_emp(e_id)
         );

     '''


if __name__ == '__main__':
    emp = [{'id': 1, 'name': 'Harry Potter', 'gender': 1, 'mgr': 1, 'salary': 15000.00, 'address': 'Hogwarts',
            'birth': '1990-01-01', 'dept': 1}, {'id': 2, 'name': 'Ronald Wesley', 'gender': 1, 'mgr': 1,
                                                'salary': 10000.00, 'address': 'Hogwarts', 'birth': '1991-02-02',
                                                'dept': 1}
           ]

    connection = get_connection()
    # insert0(connection)
    # insert(emp, connection)
    # insert2(connection)
    select(connection)
    select2()
    connection.close()
