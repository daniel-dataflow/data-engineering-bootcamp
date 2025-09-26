from threading import Thread


def bark(name, msg):
    for i in range(10):
        print(f"{name}: {msg} ({i}번지)")

if __name__ == '__main__':
    thread01 = Thread(target=bark, args=("멍멍이1", "멍멍!"))
    thread02 = Thread(target=bark, args=("멍멍이2", "멍멍!"))
    thread03 = Thread(target=bark, args=("야옹이1", "야옹!"))
    thread04 = Thread(target=bark, args=("야옹이2", "야옹!"))

    # start = run 호출
    thread01.start()
    thread02.start()
    thread03.start()
    thread04.start()

    print("끝!")

    """
    thread : process 내부의 작업 단위 - 메모장에 쓰이는 내용
    process :  program을 실행하여 memory에 실제로 적재된 구현체 (job, task) - 메모장이 실행되어 띄어진 모습
    program : (code로 이뤄어진) 실행 가능한 파일    - 메모장 실행파일
    스케줄러 : 메인와 쓰레드의 변경하는 놈
    스케줄링 : 작업을 조율해주는 행위
    컨테스트 스위칭 : 메인과 쓰레드의 바뀌는 행위를 명칭함
    """

