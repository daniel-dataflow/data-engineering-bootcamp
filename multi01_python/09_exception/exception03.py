def input_num():
    while True:
        try:
            n = int(input("숫자를 입력해 주세요 : "))
            break
        except ValueError:
            print("숫자가 아님니다!! 숫자만 다시 입력해 주새요 : ")

    return n

if __name__ == '__main__':
    a = input_num()
    b = input_num()
    print(f"{a} + {b} = {a + b}")

