from threading import Thread, Lock
from time import sleep


# thread가 공유할 자원
resource = 0

def increase_resource():
    global resource
    resource += 1

class LockTest(Thread):
    def __init__(self, name, lock):
        Thread.__init__(self)
        self.name = name
        self.lock = lock

    def run(self):
        global resource
        resource = 0

        for i in range(10):
            # lock.acquire : 잠금
            self.lock.acquire()
            increase_resource()
            sleep(0.5)
            # lock,release : 잠금 해제
            self.lock.release()
            print(f"{self.name} {i} : {resource}")

if __name__ == '__main__':
    # Lock : 특정 Thread가 resource에 접근 시, 다른 Thread는 접근 불가!
    lock = Lock()
    thread_list = [
        LockTest("t01", lock),
        LockTest("t02", lock),
        LockTest("t03", lock),
        LockTest("t04", lock),
        LockTest("t05", lock),
    ]

    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print(resource)