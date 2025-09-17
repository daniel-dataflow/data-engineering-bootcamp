def param01(param):
    print(f"parameter 안에 저장되어 있는 값 :  {param}")

def param02(a, b="default"):
    print(a)
    print(b)

# *args : arguments - > 여러 개의 값을 한 번에 받을 수 있다.(packing)
def param03(*args):
    print(args)
    for i, arg in enumerate(args):
        print(f"{i}번지의 값 : {arg}")

# **kwargs : keyword arguments -> key, value (딕셔너리로 활용된다)
def param04(**kwargs):
    print(kwargs)
    for k, v in kwargs.items():
        print(f"key={k} \t value={v}")


def param05(*args, **kwargs):
    print(args)
    print(kwargs)


if __name__ == "__main__":
    param01("arguments")

    param02(1, 2)
    param02("abc")
    param02(a="def")
    param02("aa","bb")

    param03("a", "b")
    param03(1, 2, 3)
    param03(["a", "b", "c"])
    param03(*["have", "a","nice", "day", "!!!"])

    param04(a=1, b=2, c=3)
    #param04({"a":4, "b":5, "c":6})
    #param04(*{"a": 4, "b": 5, "c": 6})
    param04(**{"a": 4, "b": 5, "c": 6})

    param05("a", "b", "c", "d", e=5, f=6, g=7)