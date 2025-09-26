from threading import Thread
from datetime import datetime


value = 100000000
result = 0


def calc(name, start, end):
    cnt = 0
    for i in range(start, end):
        cnt += 1

    print(f"{name} : {cnt}")

    global result
    result += cnt

def single_thread():
    start_time = datetime.now()

    global value
    global result

    result = 0

    t01 = Thread(target=calc,args=("single_thread", 1, value+1))
    t01.start()
    t01.join()
    print(f"result : {result}")
    end_time = datetime.now()
    print(f"single_thread runtime : {end_time - start_time}")


def multi_thread():
    start_time = datetime.now()

    global result
    result = 0

    t01 = Thread(target=calc,args=("t01", 1, value//2))
    t02 = Thread(target=calc,args=("t02", value//2, value+1))

    t01.start()
    t02.start()

    t01.join()
    t02.join()

    print(f"result : {result}")
    end_time = datetime.now()
    print(f"multi_thread runtime : {end_time - start_time}")


if __name__ == '__main__':
    single_thread()
    print("-" * 10)
    multi_thread()
    # 차이가 별로 없는 이유 : GIL때문!
    #GIL : Global Interpreter Lock


