import asyncio

async def hello(future, name):
    print(f"Hello, {name}!")
    await asyncio.sleep(3)
    future.set_result("Nice to meet you")       # return한 값과 동일하게 출력되더라.

async def main():
    loop = asyncio.get_event_loop()

    future01 = loop.create_future() # 새로운 쓰레드 생성
    future02 = loop.create_future()

    loop.create_task(hello(future01, "a"))
    loop.create_task(hello(future02, "2"))

    print(await future01)
    print(await future02)

if __name__ == "__main__":
    asyncio.run(main())