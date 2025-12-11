from pymongo import MongoClient


def connect01():
    client = MongoClient('localhost', 27017)
    db_name = client.list_database_names()
    for name in db_name:
        print(name)


def connect02():
    # protocol://ip:port
    # protocol://ip:port/batabase
    uri = "mongodb://localhost:27017"
    client = MongoClient(uri)

    db = client["test"]

    collection_names = db.list_collections()
    for collection in collection_names:
        print(collection)


if __name__ == '__main__':
    # connect01()
    connect02()
