from concurrent.futures import ThreadPoolExecutor
from threading import Thread


def return_calc():
    result = 0
    for i in range(10):
        result += i
    return result

def non_turn_calc():
    result = 0
    for i in range(10):
        result += i
    print(f"result: {result}")

def thread_way():
    # Thread
    # 하나의 객체 당 하나의 작업만 관리 가능
    # 리턴되는 결과 확인 불가
    # 작업 상태 확인 불가
    thread01 = Thread(target=return_calc)
    thread02 = Thread(target=non_turn_calc)

    thread01.start()
    thread02.start()

    thread01.join()
    thread02.join()

    print(thread01)
    print(thread02)

def threadpool_way():
    # ThreadPoolExwcutor
    # 하나의 객체로 여러 개의 작업 관리 가능
    # return 되는 값 확인 가능
    # 작업 상태(status) 확인 가능
    # 추가적인 기능들이 더 있어요!!!
    with ThreadPoolExecutor(max_workers=2) as executor:
        future01 = executor.submit(return_calc)
        future02 = executor.submit(non_turn_calc)

        print(future01.result()) # 작업한 리턴 값을 보여준다.
        print(future02.result()) # 작업한 리턴 값이 없어서 None으로 나온다.

        print(future01)
        print(future02)

if __name__ == "__main__":
    thread_way()
    print("="*10)
    threadpool_way()