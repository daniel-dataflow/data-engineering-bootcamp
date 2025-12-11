from pymongo import MongoClient


def aggre():
    client = MongoClient("mongodb://localhost:27017")
    db = client["test"]
    collection = db["score"]

    result = collection.aggregate([
        # "$group":묶어줄거다
        # "_id": "null" : 전체를
        # "average": 집계함수를 에 넣을거다.
        # "$avg": 평균으로 (2번 사용되어 개인평균, 전체평균으로 작업되었다)
        # "$kor" : 각 도큐먼트에서 가져온 값
        {"$group": {"_id": "null",
                    "average": {"$avg": {"$avg": ["$kor", "$eng", "$math"]}}}},
        # "$project": 출력하자
        # "_id":0, 출력 안함
        # "average": {"$round": ["$average", 0] 0번째 자리수로 반올림 할거다.
        # "average" : 표시되는 명칭
        # "$average" : 위에서 가져온 값
        {"$project": {"_id":0, "average": {"$round": ["$average", 0]}}}
    ])

    print(result)
    print(list(result))



if __name__ == '__main__':
    aggre()