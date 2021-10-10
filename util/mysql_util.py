import pymysql
from pymysql.cursors import DictCursor

CONFIG = {'host': 'localhost', 'port': 3306, 'user': 'cai', 'password': '123456', 'database': 'company',
          'charset': 'utf8'}  #, "cursorclass":DictCursor

class DB:

    def __init__(self, database='company'):
        CONFIG['database'] = database
        self.conn = pymysql.connect(**CONFIG)

    def __enter__(self):
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None


class BaseDao:
    def __init__(self, database='company'):
        self.db = DB(database)

    def find_all(self, table, where=None):
        sql = 'select * from %s ' % table    # can't do select * from %s, %s should be values for comparison/input,
        if where:
            sql += where
        with self.db as c:

            c.execute(sql)
            row_headers = [x[0] for x in c.description]  # this will extract row headers
            rv = [[i for i in row] for row in c.fetchall()]

            json_data = []
            for result in rv:
                for i in range(len(result)):
                    if type(result[i]) is bytes:
                        b= result[i][0]
                        result[i] = b
            for result in rv:
                json_data.append(dict(zip(row_headers, result)))
        #return json.dumps(list(json_data))
        return json_data

    def find_id(self, table, id):
        with self.db as c:
            sql = 'SELECT * from %s where d_id = ' % (table,) + '%(id)s'
            c.execute(sql, {'id': id})
        return [row for row in c.fetchall()]

    def insert(self, table, data):  #dict data
        placeholders = ', '.join(['%s'] * len(data))
        columns = ', '.join(data.keys())
        sql = "INSERT INTO %s ( %s ) VALUES ( %s )" % (table, columns, placeholders)
        with self.db as c:
            c.execute(sql, list(data.values()))
            if c.rowcount > 0:
                return True

    def insert2(self, table, **data):
        sql = "INSERT INTO %s(%s) VALUES (%s)"
        placeholders = ','.join(['%%(%s)s' % key for key in data]) #['%%(%s)s' % key for key in data]
        columns = ','.join([key for key in data])

        with self.db as c:
            c.execute(sql % (table, columns, placeholders), data)
            if c.rowcount > 0:
                return True
    def close(self):
        self.db.close()

if __name__ == '__main__':
    bd = BaseDao('company')
    #print(bd.find_all('t_emp'))
    #print(bd.find_all('t_emp', 'where e_id=1'))
    dept = dict(d_id=7, d_name='Gryffindor', d_address='Hogwarts')
    #bd.insert2('t_dept', **dept)#{'d_id':4, 'd_name':'Gryffindor'}
    print(bd.find_id('t_dept',1))
    bd.close()


