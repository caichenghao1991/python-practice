import pymysql


def get_connection():
    conn = pymysql.connect(host='localhost', port=3306, user='cai', password='123456',
                           database='company', charset='utf8')
    return conn


def insert(emp_list, conn):
    try:
        with conn.cursor() as cursor:
            for e in emp_list:
                result = cursor.execute(
                    "INSERT INTO t_emp VALUES(%s,%s,%s,%s,%s, %s,%s,%s);",
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


def main(conn):
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

    ''' 
         CREATE DATABASE company;
         USE company;
         CREATE TABLE t_dept(
            d_id INT NOT null,
            d_name VARCHAR(50) NOT NULL,
            d_address VARCHAR(100),
            INDEX i_d_name (d_name),
            PRIMARY KEY (d_id)
        );
         
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
    main(connection)
    insert(emp, connection)
