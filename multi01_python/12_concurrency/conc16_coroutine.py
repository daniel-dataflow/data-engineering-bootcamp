def heelo():
    while True:
        name = yield
        print(f"Hello, {name}")

if __name__ == '__main__':
    coroutine = heelo()
    # coroutine.__next__()
    next(coroutine)

    # send() : yield를 통해 generator에 값을 전달
    coroutine.send("홍길동")
    coroutine.send("이순신")
    coroutine.send("김선달")
