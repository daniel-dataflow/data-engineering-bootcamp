# f(x) : 함수 = 입력 -> 처리 -> 출력 : 기능

def hello01():
    print("hello, world!")
    print("hello, python!")

hello01()

def hello02():
    msg = "hello, python!!!"
    return msg

print(hello02())


def hello03():
    return {"name": "admin", "message": "hello, service!"}

print(hello03())

result = hello03()
print(f"name : {result['name']}")

def hello04():
    print("no return")
    # return 되는 값 없다. = None (리턴은 기본적으로 None 값을 가지고 있고 생략되어도 해당 값을 제시한다)
    return

print(hello04())

def hello05():
    return 1
    return 2
    return 3

print(hello05())
print(hello05) #메모리 주소를 출력한다.함수가 만들어진 메모리를 객체를 표시한다.

