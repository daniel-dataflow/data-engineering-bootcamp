"""
1
11
1101
10001
111001
1000001
11010001
101000001
1100100001
.....

주어진 수열의 n번째 수에서 약수인 숫자는 1, 아닌 숫자는 0으로 표시하는 수열 변환하는 코드를 만들어라
"""


# 주어진 수에 대해 약수인 숫자는 1, 아닌 숫자는 0으로 표시하는 수열 생성
def generate_divisor_sequence_up_to_n(n):
    result = []

    # 1부터 n까지 반복
    for number in range(1, n + 1):
        sequence = []
        for i in range(1, number + 1):
            # i가 number의 약수라면 1, 아니면 0
            if number % i == 0:
                sequence.append("1")
            else:
                sequence.append("0")

        # 결과 리스트에 수열을 추가
        result.append("".join(sequence))

    return result


# 테스트
n = 12  # 예시로 12까지 출력
result = generate_divisor_sequence_up_to_n(n)

# 결과 출력
for seq in result:
    print(seq)


