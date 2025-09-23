class A:
    pass

class CreateMeta(type):
    def __new__(cls, *args, **kwargs):
        print("metaclass new")
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        print("metaclass init")

    # callable object로 만들어준다!
    def __call__(self, *args, **kwargs):
        print("metaclass call")
        return super().__call__()

if __name__ == "__main__":
        num = 10
        print(type(num))
        print(f"그래서 그 클래스의 타입은 뭔데 :{type(type(num))}")

        print(type(A))
        # type : 값의 형태다
        # "A라는 class"라는 값의 형태 : type
        # = "A라는 class"라는 인스턴스의 타입이 type
        # class도 객체이다.

        # type을 이용하여 class를 생성
        # type  = class를 생성하는 class = metaclass
        hello_type = type("HelloWorld", (), {})
        # hello_type = HelloWorld 라는 class!!!
        print(hello_type)
        # hello_type을 instance화 하니까 객체가 만들어지더라!
        # = hello_type이 class였어!
        hello = hello_type() # 클래스가 생성된다.
        print(hello) # 주소 값이 나온다

        print("-----------------------")

        metaclass = CreateMeta("MyClass",(),{}) #__init__ ->__new__ ->
        print(metaclass)
        instance = metaclass() #객체가 만들어짐 __call__라는 녀석을 만들
        print(instance)
