from multiprocessing.process import parent_process


class Parent:
# class Parent(Object) :

    # 객체 생성( __init__가 호출)
    def __new__(cls, *args, **kwargs):
        print("parent new")

        # super() : 부모 객체
        return super().__new__(cls) # Object 의 바라보고 있다. 반드시 작성되어야 자식이 사용할 수 있다.

    # 변수 초기화
    def __init__(self):
        print("parent init")
        self.name = "parent"
        self.age = 100

    def prn01(self):
        print(f"print01: {self.name}")

    def prn02(self):
        print(f"print02: {self.name} / {self.age}")


# Parent를 상속 받은 Child class
class Child(Parent):

    def __new__(cls, *args, **kwargs):
        print("child new")
        return super().__new__(cls) # 부모를 바라보고 있다.(반드시)

    def __init__(self):
        print("child init")
        self.name = "child"

if __name__ == "__main__":
    parent = Parent() # parent init 호출 --> parent new 호출, 프린트 -> parent init 프린트
    parent.prn01()
    parent.prn02()

    print("---------")
    child = Child() # child init 호출 --> child new 호출 프린트 -> parent new 호출 프린트 -> child init 프린트
    child.prn01()
    child.prn02() # parent의 객체(instance)는 가져올 수 없다. 클래스에 있는 만 가져다 쓸 수 있다.-> self.name 이 작성되어 있지 않아서 에러가 난다.



