from redis import Redis

# decode_responses=True : --raw 와 같은 기능이다. 바이트로 저장되어 있는 값을 decoding 일반적인 문자열로 보여줄 수 있다.
r = Redis(host='localhost', port=6379, decode_responses=True)

# 작성하고
r.set("language", "python")
# 다시 가져오자
print(r.get("language"))

