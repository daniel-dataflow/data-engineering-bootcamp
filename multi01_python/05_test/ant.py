''' 
출력 예)
1 
11 
12 
1121
122111 
112213
'''

"""
i       : 1 2  3 
o(문자)  : 1 11 12
count   : 1 2

"""

def ant(i: int) -> str:
    # 이전 stage의 수
    input_num = str(i)


    # 처음 나온 수 (비교를 위한 기준)
    target = input_num[0]
    # "비교를 위한 기준의 수" 에 대한 카운트
    cnt = 0

    # 결과
    res = ""


    # 반복해서 하나씩 비교 => String = Character의 집합
    for char in input_num:
        # char와 target을 비교 -> 둘이 같다면 counting한다.
        if char == target:
            cnt += 1
        else:
            # res (결과)에 현재 target(기준) + cnt(카운팅 한 갯수)
            res += target + str(cnt)

            # 새로 나온 숫자를 기준으로 변경
            target  = char
            # 새로운 기준에 대한 카운팅
            cnt = 1
    # 가장 마지막에 나온 target에 대한 결과 추가
    res += target + str(cnt)

    return res





if __name__ == '__main__':
    n = int(input('stage : '))

    # stage 1
    print("1")

    # stage 2
    val = ant(1)
    print(val)

    # stage 3 - 스테이지2에서 나온 값을 다시 넣어주고 있다.
    for _ in range(1, n-1):
        val = ant(val)
        print(val)
