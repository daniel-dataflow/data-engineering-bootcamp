class HelloStr:
    def __init__(self, name="홍길동", age=100):
        self.name = name
        self.age = age

    # __str__ : 객체의 주소값 리턴
    # override :  부모의 속성/기능 을 자식이 재정의 (상속)
    def __str__(self):
        return f"name : {self.name}, age : {self.age}"




if __name__ == "__main__":
    class04 = HelloStr(name="이순신", age=10)
    # print는 __str__를 찾아서 프린트 해준다.
    print(class04)