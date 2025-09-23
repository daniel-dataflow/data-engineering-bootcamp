from curses import wrapper

# 3) "world" 전달
def greeting(name):
    # 4) myfunc() 전달
    def wrapper(func):
        # 5)
        def prn():
            print(f"hello {name} !!")
            # 6)
            func()
        return prn
    return wrapper

# 2)
@greeting("world")
def myfunc():
    print("hello python !!")

if __name__ == "__main__":
    # 1)
    myfunc()
    # greeting("world")(myfunc)()       #@greeting("world")을 주석할 경우 실행방법
    # => wrapper(myfunc)()
    # => prn()