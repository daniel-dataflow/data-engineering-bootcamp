from setuptools._static import Static


class Parent():

    class_val = "class variable"

    @classmethod
    def class_prn(cls): # classmethod를 넣으면 cls를 사용할 수 있고  cls를 넣은 경우 자식이 부를 경우 자식 클래스가 넘어온다.
        print(cls.class_val)

    @staticmethod
    def static_prn(): # staticmethod는 함수처럼 실행되고 만다.
        print(Parent.class_val)


class Child(Parent):
    class_val = "child's class variable"


if __name__ == '__main__':
    parent = Parent()
    parent.class_prn()
    parent.static_prn()
    print("--------")

    child = Child()
    child.class_prn()
    child.static_prn()
