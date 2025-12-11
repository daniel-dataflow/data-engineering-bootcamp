from pymongo import MongoClient

def insert01(db):
    # db.mongopy.insertOne()
    result = db.mongopy.insert_one({"name":"이동헌", "kor": 60, "eng": 65, "math": 70})
    print(result.inserted_id)

def find(db):
    for doc in db.mongopy.find():
        print(doc)


def insert02(db):
    friends = [
        {"name" : "hong-gd", "kor": 100, "eng": 95, "math": 90},
        {"name" : "kim-sd", "kor": 90, "eng": 85, "math": 80}
    ]

    result = db.mongopy.insert_many(friends)
    # insert_many 인 경우는 inserted_ids를 사용한다
    print(result.inserted_ids)



if __name__ == '__main__':
    # "mongodb://localhost:27017" : 기본적으로 잡아주지만 작성하는 것을 습관화
    client = MongoClient("mongodb://localhost:27017")
    db = client.test

    # insert01(db)
    insert02(db)

    find(db)


    client.close()
