import asyncio

# coroutine
# 함수 앞에 async
async def hello(name):
    print(f"Hello, {name}")
    # 비 동기적으로 실행하고 싶은 객체 앞에다가 await
    # async 안에 sleep 구현이 되어 있다
    await asyncio.sleep(3)
    print("Nice to meet you")

if __name__ == '__main__':
    # 동기적으로 실행
    asyncio.run(hello("async"))
    asyncio.run(hello("coroutine"))