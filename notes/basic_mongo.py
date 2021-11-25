"""
    pip install pymongo


"""
import pprint

import pymongo
import datetime
# create/acess database and collection
client = pymongo.MongoClient("mongodb://localhost:27017/")   # create a MongoClient mongod instance
    # client = MongoClient('localhost', 27017)
dblist = client.list_database_names()   # get all databases as list
print(dblist)
mydb = client['hogwarts']   # create or access the database
mycollection = mydb['student']  # create or access collection
collist = mydb.list_collection_names()  # list all collections
print(collist)
# mycollection.drop()   # drop collection

# insert document
data = {"name": "Harry", "age": 10, "courses": ["Magic Defense", "Magic Spell"], "date": datetime.datetime.now()}
    #  add "_id": 1,   if manually specify unique id, not using default ObjectId
mycollection = mydb['student']
res = mycollection.insert_one(data)  # insert data and return pymongo.results.InsertOneResult
print(res, res.inserted_id)    # return objectId


data_bulk = [{"name": "Harry", "age": 10, "courses": ["Magic Defense", "Magic Spell"], "date": datetime.datetime.now()},
             {"name": "Harry", "age": 10, "courses": ["Magic Defense", "Magic Spell"], "date": datetime.datetime.now()}]
res2 = mycollection.insert_many(data_bulk)  # insert multiple objects. can't insert same object with same reference twice
print(res2, res2.inserted_ids)  # return list of ObjectId

# update document
res3 = mycollection.update_one({"name": "Harry"}, {'$set': {"age": 11}})
res3 = mycollection.update_many({"name": "Harry"}, {'$set': {"age": 11}})
print(res3.modified_count)

# find document
data_ = mycollection.find_one({'name': 'Harry'})  # find one document
data_ = mycollection.find_one({'_id': res.inserted_id})
data_ = client.hogwarts.student.find_one({'_id': res.inserted_id})

data_cursor = mycollection.find({'name': 'Harry'})  # find multiple document, return iterable cursor
    # all = mycollection.find()  # find all data in collection
data_cursor = mycollection.find({'data': {'$gt': datetime.datetime(2009,11,12)}, 'name': 'Harry'}).sort('name')
    # range queries, multiple and conditions, sort on field.   sort('name', -1)  desc
data_cursor = mycollection.find({'name': 'Harry'},{ "_id": 0, "name": 1, "courses": 1})
    # return only selected fields, default not included except _id
data_cursor = mycollection.find({'name': {'$regex': '^H'}}).limit(3)  # regex search name start with H, limit 3 document

for doc in data_cursor:
    print(doc)

# count document
print(mycollection.count_documents({}))
print(mycollection.count_documents({'name': 'Harry'}))

# delete document
res = mycollection.delete_one({'name': 'Harry'})
print(res.deleted_count, res.acknowledged)  # 1  True
res = mycollection.delete_many({'name': 'Harry'})
    # mycollection.delete_many({})  # delete all documents


# create index
res = client.hogwarts.student.create_index([('name', pymongo.ASCENDING),('_id', pymongo.DESCENDING)], unique=True)
    # raise exception if multiple documents with same index, must unique identify document
sorted(list(mycollection.index_information()))
