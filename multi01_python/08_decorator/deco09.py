# 3) 생성한다.
class Greeting:
    # 4) 생성했기 때문에 init 안에 함수 전달
    def __init__(self, func):
        self.func = func # 객체가 가질수 있는 변수를 저장한다.

    # callable object : 객체를 함수처럼 호출
    def __call__(self, *args, **kwargs):
        print("hello, ", end="")
        self.func(*args, **kwargs)

# 2)
@Greeting
def myfunc():
    print("wolrd!")


if __name__ == "__main__":
    # 1)
    myfunc()
    # Greeting(myfunc)() # 데코레이터 없을 경우에는 이렇게 실행
    # class를 ()를 붙여서 객체화 한뒤 __call__로 함수화로 변경