class singleton(type):
    # instance 주소 저장
    __instance = {}

    # __call__ : class를 함수처럼 호출할 수 있도록 만든다.
    # -> callable object
    def __call__(self, *args, **kwargs):
        if self not in self.__instance:
            self.__instance[self] = super().__call__(*args, **kwargs)
        return self.__instance[self]

# singleton: 메모리에 객체가 단 한개만 존재하게 만드는 것
class MyClass(metaclass=singleton):
    pass

if __name__ == '__main__':
    a = MyClass() # class를 객체로 바꿀때 __call__을 사용해야 한다.
    b = MyClass()

    print(a)
    print(b)

    print(a==b)


