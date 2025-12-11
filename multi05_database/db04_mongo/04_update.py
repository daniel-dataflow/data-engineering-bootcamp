from pymongo import MongoClient

def update01(db):
    result = db.mongopy.update_one({"name": "이동헌"},{"$set": {"name":"lee-dh"}})

    print(result.matched_count)
    print(result.modified_count)


def update02(db):
    result = db.mongopy.update_many({"name": {"$regex" : "e"}},
                                    {"$set": {"kor": 0}})
    print(result.matched_count)
    print(result.modified_count)


def find(db):
    for doc in db.mongopy.find({}, {"_id":0}):
        print(doc)


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.test


    # update01(db)
    update02(db)


    find(db)


    client.close()