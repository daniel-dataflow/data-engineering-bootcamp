from datetime import datetime
from time import sleep


def mylogger(func):

    def logging():
        print(f"[logger] : {datetime.now()}")
        func()
        print(f"[logger] : {datetime.now()}")

    return logging


@mylogger
def greeting():
    sleep(2)
    print("hello, world!")


if __name__ == "__main__":
    greeting()