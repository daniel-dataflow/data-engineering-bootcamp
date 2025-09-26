def generator01(end):
    yield list(range(end))

def generator02(end):
    # yield from : 다른 generator 호출
    yield from list(range(end)) #from 가 있으면 list가 생성되지 않고 건너띄고 range와 연결된다.


if __name__ == '__main__':
    gen01 = generator01(10)
    for item in gen01:
        print(item)

    print("-"*10)
    gen02 = generator02(10)
    for item in gen02:
        print(item)