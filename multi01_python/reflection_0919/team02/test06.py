"""
n>1
a(n) = 3 * n
b(n) = b(n-1) + a(n) # n이 5의 배수가 아닐때
       b(n-1) - a(n) # n이 5의 배수일때
b(1) = 1 이라할때
b(15) 를 구하라
"""

def a(n):
    return 3 * n

def b(n):
    # b(1)의 초기값 설정
    if n == 1:
        return 1
    else:
        # n이 5의 배수일 때 b(n) 규칙에 맞추어 계산
        if n % 5 == 0:
            return b(n-1) - a(n)
        else:
            return b(n-1) + a(n)

# b(15) 구하기
result = b(15)
print(result)
