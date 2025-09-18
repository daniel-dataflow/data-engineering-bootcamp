# 전역변수
x = 10

def test01():
    print(x)

def test02():
    # 지역변수
    x = 20
    print(x)

def test03():
    # 전역변수 x를  사용하고 싶어...
    global x
    x = 30
    print(x)

test01()
test02()
test03()

print(x)

print("----------------------")

def test04():
    global y
    y = 3
    print(y)

test04()
print(y)


# shadowing : 3개의 y는 모두 다른 y
def outer01():
    y = 6

    def inner01():
        y = 9
        print(f"inner y : {y}")

    inner01()
    print(f"outer y : {y}")

outer01()
print(f"gobal y : {y}")
print("----------------------")


def outer02():
    global y
    y = 6

    def inner02():
        y = 9
        print(f"inner y : {y}")

    inner02()
    print(f"outer y : {y}")

outer02()
print(f"gobal y : {y}")
print("----------------------")


def outer03():
    y = 9

    def inner03():
        nonlocal y # 나를 감싸고 있는 녀석까지 대치해서 사용하겠다.
        y = 3
        print(f"inner y : {y}")
        def inner_inner():
            nonlocal y
            y = 1
            # print(y)
        inner_inner()
        print(f"inner_inner y : {y}")

    inner03()
    print(f"outer y : {y}")

outer03()
print(f"gobal y : {y}")
print("----------------------")

# python namespace - 추가적으로 공부하고 싶다면

