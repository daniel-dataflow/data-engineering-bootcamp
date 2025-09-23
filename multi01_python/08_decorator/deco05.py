# 3) 7) 여러개 들어가도 상관 없다.
def greeting(func):

    def prn(*args, **kwargs):
        print("hello ", end="")
        func(*args, **kwargs)

    # 4) 8)
    return prn

# 2)
@greeting
def myfunc01(name):
    print(name)

# 6)
@greeting
def myfunc02():
    print("python!!!")




if __name__ == "__main__":
    # 1) 5)
    myfunc01("DaeSung")
    myfunc02()



