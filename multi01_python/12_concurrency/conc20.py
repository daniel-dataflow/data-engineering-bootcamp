import asyncio

# coroutine
# 함수 앞에 async
async def hello(name):
    print(f"Hello, {name}")
    await asyncio.sleep(3)
    print("Nice to meet you")

async def main():
    # coroutine
    await hello("coroutine")

    # coroutine -> task
    hello01 = asyncio.create_task(hello("async"))
    hello02 = asyncio.create_task(hello("coroutine"))

    # 비 동기적으로 실행!
    await hello01
    await hello02
    # awaitable :  await 할 수 있는 객체
    # coroutine, task, future,  __await__() 구현 (반드시 iterator를 리턴)
    print(hello01)
    print(dir(hello01))

if __name__ == '__main__':
    # 비동기적 실행
    # 고수준 실행
    asyncio.run(main())
