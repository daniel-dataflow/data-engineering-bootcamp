class MyEnumerate:

    @staticmethod
    def my_enumerate(iterable):
        # hint! yield를 사용하자.
        # 결과 :  enumerate(["python", "numpy", "pandas"]) 과 동일하게 나와야 한다.
        idx = 0
        for item in iterable:
            yield idx, item
            idx += 1




if __name__ == '__main__':
    subject = MyEnumerate.my_enumerate(["python", "numpy", "pandas"])
    print(subject)
    print(dict(subject))
    print(list(subject))