from datetime import datetime
from time import sleep

def mylogger(name):
    def wrapper(func):
        def logging():
            print(f"{name} : {datetime.now()}")
            func()
            print(f"{name} : {datetime.now()}")
        return logging
    return wrapper

@mylogger("DaeSung)")
def greeting():
    sleep(2)
    print("hello, world!")

@mylogger("hong-gd")
def goodbye():
    sleep(2)
    print("bye, world!")



if __name__ == "__main__":
    greeting()
    print("-" * 15)
    goodbye()



