import asyncio

# coroutine
# 함수 앞에 async
async def hello(name):
    print(f"Hello, {name}")
    await asyncio.sleep(3)
    print("Nice to meet you")

async def main():
    # await 을 사용함으로써 함수호출을 coroutine로 변경해줄 수 있다.(메인과 같이 진행되는)
    await hello("coroutine")

    # coroutine -> task
    hello01 = asyncio.create_task(hello("async"))
    hello02 = asyncio.create_task(hello("coroutine"))

    await hello01
    await hello02
    print(hello01)
    print(dir(hello01))

if __name__ == '__main__':
    # 비동기적 실행
    # 고수준(추상적이다. 추상화 잘되어 있다.) 실행
    # asyncio.run(main())

    # 저수준 실행 - 여러가지 단계를 거쳐 진행해야 한다
    # 하지만 더 많은 기능을 사용
    # new_event_loop() :  event loop를 생성!
    # evnet loop :  전체적인 조율/처리
    loop = asyncio.new_event_loop() # 여러가지 실행되는 것을 제어해 줄것이다.
    loop.run_until_complete(main()) # 메인이 끝날때까지 실행해라
    loop.close() # 작업이 다 완료 되었으면 loop도 종료하겠다.
