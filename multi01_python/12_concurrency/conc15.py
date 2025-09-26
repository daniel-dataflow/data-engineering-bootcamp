from concurrent.futures import ProcessPoolExecutor
from time import sleep


values = list(range(1, 1001))

def power3(n):
    return n ** 3

def work():
    print("work start")
    sleep(0.5)
    print("work stop")

def callback(future):
    print(f"future.running: {future.running()}")
    print(f"future.done: {future.done()}")

def threadpool():
    with ProcessPoolExecutor(max_workers=2) as executor:
        # map : 각각의 하나 하나에 값을 가져와서 함수를 실행하는 역할
        result = executor.map(power3, values)
        print(list(result))

        future = executor.submit(work)
        # callback:
        # add_done_callback() : future가 완료 된 후 arguments로 받은 함수 호출!
        # -> future 객체가 전달된다.
        future.add_done_callback(callback)

if __name__ == "__main__":
    threadpool()

