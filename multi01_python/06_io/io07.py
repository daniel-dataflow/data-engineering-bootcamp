import pickle

score = {"name": "hong-gd", "kor": 100, "eng": 67, "math" : 89}

with open("test02.txt", "wb") as file:
    pickle.dump(score, file)

"""
pickling (dump) : 객체 -> 파일
unpicking (load) : 파일 -> 객체
"""