from pymongo import MongoClient


def find01(db):
    socre = db["score"]
    print(socre)
    # cursor 객체로 가져온다.
    print(socre.find())
    for doc in socre.find():
        print(doc)


def find02(db):
    score = db.score
    # 파이썬에서는 ""를 안 붙이면 변수로 인식 ""붙이면 필드명으로 인식할 수 있다.
    for doc in score.find({"kor" : {"$gte" : 80}}):
        print(doc)


def find03(db):
    # score = db["score"]
    score = db.score
    for doc in score.find({"kor" : {"$gte" : 80}}, {"_id":0, "name":1, "kor":1}):
        print(doc)



if __name__ == '__main__':
        client = MongoClient()
        # db = client["test"]
        db = client.test


        # find01(db)
        # find02(db)
        find03(db)

        client.close()



