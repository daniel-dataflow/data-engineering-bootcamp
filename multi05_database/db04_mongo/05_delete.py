from pymongo import MongoClient


def delete01(db):
    result = db.mongopy.delete_one({"name": "lee-dh"})
    print(result.deleted_count)


def delete02(db):
    result = db.mongopy.delete_many({"name": {"$regex": "d$"}})
    print(result.deleted_count)




def find(db):
    for doc in db.mongopy.find({}, {"_id": 0, "name":1} ):
        print(doc)




if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017")
    db = client["test"]


    # delete01(db)
    delete02(db)

    find(db)


    client.close()