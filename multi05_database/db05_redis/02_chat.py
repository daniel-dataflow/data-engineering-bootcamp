import redis
import threading
from time import sleep


r = redis.Redis(host='localhost', port=6379, decode_responses=True)
channel_name = 'chat'
# threading.event : true/false 값을 가진다. (초기 상태는 false)
# thread가 set() 호출 할 때 까지 다른 thread는 wait 상태가 된다.
# 우리는 listener 가 subscribe 할 때 까지 publisher가 대기하게 만들려고 사용
subscribed_event = threading.Event()


def listener():
    pubsub = r.pubsub()
    pubsub.subscribe(channel_name)

    for message in pubsub.listen():
        # print(message)
        if message["type"] == "subscribe":
            print("[구독]")
            subscribed_event.set()

        if message["type"] == "message":
            print(f"[수신] : {message['data']}", flush=True)


def publisher():
    print("메시지를 입력하세요. ('q'를 입력하면 종료)")
    while True:
        try:
            sleep(0.1)
            msg = input("[발신] : ")
            if msg == 'q':
                print("[종료]")
                break

            r.publish(channel_name, msg)


        except KeyboardInterrupt:
            print("[종료]")
            break

        except Exception as e:
            print(f"[오류] : {e}")
            break


if __name__ == '__main__':
    threading.Thread(target=listener, daemon=True).start()

    subscribed_event.wait()

    publisher()

    """
    wsl -> redis-cli -> subscribe chat 하면 발신 내용을 구독받아서 볼 수 있다
    = subscribe만 하면 scale out 가능! 
    """
