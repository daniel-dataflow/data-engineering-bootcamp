# 3) func에 def myfunc(name):가 통채로 들어간다.
def greeting(func):

    # 4)
    def prn(name):
        print("Hello, ", end="")
        # 5)
        func(name)
    # 6)
    return prn

# 2)
@greeting
def myfunc(name):
    print(name)

if __name__ == "__main__":
    # greeting(myfunc)("python!")
    # 1)
    # 7)
    myfunc("python!")