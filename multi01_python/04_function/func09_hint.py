# tupe annotation type
# type hint

# 함수의 return 값이 있는 경우, 해당 값의 type을 설명
def test01(a, b) -> int:
    x, y = int(a), int(b)
    return x + y


# parameter의  type 설명
def test02(a: int, b: str) -> str:
    result = str(a) + b
    return result

# parameter : type = default
def test03(a: int=0, b: int=0) -> int:
    return a + b

if __name__ == "__main__":
    # 변수의 type 명시
    c : int = 1
    print(test01(c, 2))
    print(test02(1, "2"))
    print(test03(1))

    # pep484 <- 파이썬 공식문서에서 확인 할 것
    # Fast API일때 적극적으로 사용됨