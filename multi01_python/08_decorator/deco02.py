# 2)
def greeting(func):
    def prn():
        print("hello, ", end="")
        # 3)
        func()
     # 5)
    return prn

# 4)
def myfunc():
    print("world")

if __name__ == "__main__":
    # 1)
    greeting(myfunc)()