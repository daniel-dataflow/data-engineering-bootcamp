from enum import Enum



class Color(Enum):
    RED = 1
    BLUE = 2
    GREEN = 3

# enum : 상수 저장(변하지 않는 수)
if __name__ == '__main__':
    print(Color)
    print(Color.RED)
    print(Color.RED.name)
    print(Color.RED.value)

    print(Color.__iter__())
    print(list(Color.__iter__())) # enum을 리스트로 만들때 해당 값은 <>안에 넣어서 표현한다.
