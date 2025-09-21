# 2개의 자연수를 입력받아 최대공약수와 최대공배수를 구하는 함수를 구현하세요
# recursion 활용
# *Euclidean algorithm(유클리드 호제법)* 검색 시 쉽게 구현가능!

"""
자연수를 입력하세요: 10
또다른 자연수를 입력하세요: 20
최대공약수: 10
최소공배수: 20
"""

def min_devisor(i: int, j: int) -> int:
    # recursion 활용
    gcd_val = max_multiple(i, j)

    # 최소공배수 계산: (두 수의 곱) / 최대공약수
    # // 연산자로 정수 나눗셈을 보장합니다.
    lcm_val = (i * j) // gcd_val

    # 계산된 최소공배수 '값'을 반환합니다.
    return lcm_val


def max_multiple(i: int, j: int) -> int:
    # 유클리드 호제법을 사용하여 최대공약수를 계산합니다.
    while j >0 :
        i, j = j, i % j
    return i

def print_result(i: int, j: int) -> None:
    gcd_result = max_multiple(i, j)
    lcm_result = min_devisor(i, j)

    print(f"최대공약수: {gcd_result}")
    print(f"최소공배수: {lcm_result}")


if __name__ == "__main__":
    a = int(input("자연수를 입력하세요: "))
    b = int(input("또다른 자연수를 입력하세요: "))
    print_result(a, b)