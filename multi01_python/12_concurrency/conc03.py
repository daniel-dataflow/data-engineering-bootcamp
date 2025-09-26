from threading import Thread
from time import sleep


value = 10

def calc(name, num, sleep_time):
    global value
    for i in range(10):
        value = value + num
        print(f"{name}{i} : {value}")
        sleep(sleep_time)


if __name__ == '__main__':
    print("main thread start!")
    test = Thread(target=calc, args=("test",1,0.5))
    deamon_bk = Thread(target=calc, args=("deamon_bk",-1,1))

    # deamon.setDaemon(True)
    deamon_bk.daemon = True
    # deamon thread : thread를 보조하는 thread(deamon이 아닌 다른 thread가 더 이상 작업이 없을 때 종료)
    # deamon 은 주로 백그라운드 작업을 실행하게 만든다.

    test.start()
    deamon_bk.start()

    print(f"test.isDaemon: {test.daemon}")
    print(f"daemon.isDaemon: {deamon_bk.daemon}")
    # print(f"test.isDaemon: {test.isDaemon()}")

    print("main thread end!!!")