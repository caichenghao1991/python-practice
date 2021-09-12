import redis


def print_bin(*args):
    for _ in args:
        if type(_) is bytes:
            print(_.decode())   # print bytes data
        else:
            try:
                iterator = iter(_)
                print([x.decode() for x in _])   # print list of bytes data
            except TypeError:
                print(_)       # print normal data


def main():
    c = redis.Redis(host='127.0.0.1', port=6379, password=123456)  # password depend on server requirepass

    c.flushdb()  # clear database

    # set
    c.set('user1', 'Harry', ex=300)  # key - value ttl
    print_bin(c.get('user1'))  # b'Harry' if no decode  Harry
    print(c.ttl('user1'))  # time to live  300
    c.mset({'user2': 'Ronald', 'user3': 'Hermione', 'score': 12})  # add multiple to set
    c.append('user2', ' Wesley')  # append to value for key
    print_bin(c.mget('user2', 'user3'))  # ['Ronald Wesley', 'Hermione']  get multiple value
    print_bin(c.getrange('user1', 0, 2))  # Har  get substring of value
    c.incr('score')  # add 1 to score value  increase 1 for value
    print_bin(c.get('score'))  # 13
    c.incrby('score', 10)  # add 10 to score value  increase 10 for value
    print_bin(c.get('score'))  # 23
    c.delete('score')
    print(c.exists('score'))  # 0 if not exist

    # hash
    c.hset('student', 'id', 1)  # create hash add 1 k-v to hash
    c.hmset('student', {'name': 'Harry Potter', 'house': 'Gryffindor'})  # add multiple k-v to hash
    print_bin(c.hget('student', 'id'))  # get hash key related value
    print_bin(c.hmget('student', ['name', 'house']))  # get multiple hash key related value
    print_bin(c.hkeys('student'), c.hvals('student'))  # get keys / values
    print_bin(c.keys('*'))  # search all keys start with s
    c.hdel('student', 'house')  # delete key
    print_bin(c.hgetall('student'))  # get keys

    # list
    c.lpush('list1', 10, 20, 30, 40, 50)  # add at front 10, 20...
    print_bin(c.lrange('list1', 0, -1))
    c.lpush('list1', 10, 20, 30, 40, 50)
    print(c.llen('list1'))
    c.lpop('list1')
    print_bin(c.lindex('list1', 0))  # 40

    # set
    c.sadd('set1', 10, 20, 30, 20, 30, 40, 50)  # add values to set
    c.sadd('set2', 30, 40, 60)  # add values to set
    c.srem('set1', 50)  # remove member in set
    print_bin(c.smembers('set1'))  # ['30', '40', '10', '20'] return all members
    print(c.scard('set1'))  # 4 return members count
    print(c.sismember('set1', 10))  # True  check members in set
    print_bin(c.sinter('set1', 'set2'))  # ['40', '30'] intersection of sets
    print_bin(c.sunion('set1', 'set2'))  # ['20', '30', '40', '10', '60']  union of sets
    print_bin(c.sdiff('set1', 'set2'))  # ['20', '10'] items only in set1

    # zset (ordered via key)
    c.zadd('score', {'a': 10, 'b': 50, 'c': 30})  # sort via int value
    print_bin(c.zrange('score', 0, -1))  # ['a', 'c', 'b']
    c.zincrby('score', 50, 'a')
    print_bin(c.zrevrange('score', 0, -1))  # ['a', 'b', 'c']

    print(c.dbsize())  # total keys
    print_bin(c.keys('*'))  # search all keys     *s start with s
    c.close()


if __name__ == '__main__':
    main()
