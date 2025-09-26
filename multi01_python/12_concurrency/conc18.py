def coroutine_sub():
    character = list()

    while True:
        char = yield

        if char is None:
            return character

        character.append(list(map(lambda x: ord(x), char))) # ord : 해당문자에 대응되는 유니코드를 준다.

def coroutine():
    while True:
        character = yield from coroutine_sub() # yield from : 제너레이터가 다른 제너레이터에 요청을한다.
        print(character)

if __name__ == '__main__':
    my_char = coroutine()
    next(my_char)

    # 문자가 ord로 전달된 유니코드
    my_char.send("Han")
    my_char.send("Dae")
    my_char.send("Sung")

    my_char.send(None)