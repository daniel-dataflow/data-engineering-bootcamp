import asyncio

async def mysum():
    print("start")
    result = 0
    for i in range(1, 101):
        result += i
    await asyncio.sleep(3)
    print("end")
    return result

async def main():
    result = await asyncio.gather(mysum(), mysum(), mysum()) # 비동기적으로 진행할 것인데 gather: 모아서 실행할 수 있도록
    return result

if __name__ == "__main__":
    print(asyncio.run(main()))