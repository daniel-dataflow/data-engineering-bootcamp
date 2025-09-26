from threading import Thread
from multiprocessing import Process
from datetime import datetime

def work():
    pass


if __name__ == '__main__':
    workers = 10

    thread_start_time = datetime.now()

    threads = [Thread(target=work) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    thread_end_time = datetime.now()
    print(f"thread: {thread_end_time - thread_start_time}")



    process_start_time = datetime.now()

    process_list = [Process(target=work) for _ in range(workers)]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()

    process_end_time = datetime.now()
    print(f"process: {process_end_time - process_start_time}")

    # Thread 가 더빠르다.