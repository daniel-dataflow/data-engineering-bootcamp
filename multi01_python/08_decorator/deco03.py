def greeting(func):

    def prn():
        print("hello, " , end="")
        func()

    return prn

# decorator로 만들면 greeting이라는 함수에 myfunc가 값으로 전달!
@greeting #골팽이 그래팅을 만나면 함수나 클래스 이름이 될텐데 데코레이터가 걸려 있는 녀석에서 전달해서 리턴한 함수를 실행하게 될거야.
def myfunc():
    print("world")

if __name__ == "__main__":
    # greeting(myfunc)() 이것을 편하게 쓰기 위해서 데코레이터를 사용하게 되었다.
    myfunc()

