# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import pymysql
import redis
from scrapy.exceptions import DropItem


class RedisPipeline(object):
    def open_spider(self, spider):
        self.r = redis.Redis(host='127.0.0.1', port=6379,  db=2)  #password=123456,

    def close_spider(self, spider):
        self.r.close()
    def process_item(self, item, spider):
        if self.r.sadd('dishes', item['d_name']):
            return item
        raise DropItem

class ScrapyBasicPipeline:
    def open_spider(self, spider):
        self.conn = pymysql.connect(host='127.0.0.1', port=3306, db='company',
                                    user='cai', password='123456')
        self.cur = self.conn.cursor()
        #self.cur.execute('delete from t_dish')
        self.conn.commit()

    def close_spider(self, spider):
        self.cur.close()
        self.conn.close()

    def process_item(self, item, spider):
        keys = item.keys()
        values = [item[k] for k in keys]
        sql = "insert into t_dish ({}) values ({})".format(','.join(keys), ','.join(['%s'] * len(values)))
        self.cur.execute(sql, values)
        self.conn.commit()
        return item
