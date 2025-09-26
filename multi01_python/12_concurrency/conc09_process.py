from multiprocessing import Process
import os


def child():
    # pid : process id (os가 내부적으로 process를 식별하기 위해 사용)
    print(f"child : {os.getpid()}")


if __name__ == '__main__':
    print(f"parent : {os.getpid()}")
    children = [Process(target=child) for _ in range(3)]

    for prc in children:
        prc.start()
