from threading import Thread
from multiprocessing import Process
from datetime import datetime


# conc07.py 복사
def mylogger(func):
    def logging():
        start = datetime.now()
        func()
        end = datetime.now()
        print(f"[time] :  {end - start}")
    return logging

def calc(name):
    result = 0

    for i in range(100000000):
        result += i

    with open(f"{name}03.txt", "a") as f:
        f.write(str(result) +"\n")

@mylogger
def func_way():
    for i in range(10):
        calc("func")

@mylogger
def thread_way():
    threads = [Thread(target=calc, args=("thread", )) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

@mylogger
def process_way():
    process_list = [Process(target=calc, args=("process", )) for _ in range(10)]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

if __name__ == "__main__":
    func_way()      # cpu 연산에선 쓰레드와 비슷한 속도가 난다.
    thread_way()    # io/ 네트워크 작업일 때 유리
    process_way()   # cpu 작업이 많을 때 유리

