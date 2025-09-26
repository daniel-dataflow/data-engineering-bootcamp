from multiprocessing import Process
from time import sleep


def deamon_process():
    while True:
        print("❤️️")
        sleep(1)

def worker_process():
    print("work start")
    deamon = Process(target=deamon_process) # 새로운 데몬 프로세서 만들어줘.
    deamon.daemon = True
    deamon.start()
    sleep(5)
    print("work stop")

if __name__ == '__main__':
    worker_process()
