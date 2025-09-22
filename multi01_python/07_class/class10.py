class Parent:
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *arg, **kwargs)

    def __init__(self):
        print("Parent init")
        self.name = "parent"
        self.age = 10

    def prn01(self):
        print(f"Parent prn01: {self.name}")

    def prn02(self):
        print(f"Parent prn02: {self.name} / {self.age}")

class Child(Parent):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *arg, **kwargs)

    def __init__(self):
        super().__init__() # 이 구문을 통해서 부모가 가지고 있는 변수 초기화
        print("Child init")
        self.name = "child"

if __name__ == "__main__":
    child = Child() # Child()라고 쓰면 객체화 되고-> child init 호출 --> child new 호출  -> parent init 호출 -> parent new 호출
                    # -> Object -> 리턴으로 부모객체 만들어짐 -> 리턴으로 자식객체가 만들어짐 -> 인스턴스 함수를 초기화
                    # -> super-> parent init 호출 프린트-> child init -> 출력
    child.prn01()
    child.prn02()

