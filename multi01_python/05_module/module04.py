# import math
# import math as m
# as : alias (별칭)
from math import pi #pi만 가져와서 쓸거야

def circle(r):
    # return math.pi * r * r
    # return m.pi * r * r
    return pi * r * r

if __name__ == '__main__':
    r = input("반지름 입력 : ")
    print(f"반지름이 {r} 인 원의 넓이는 {circle(int(r))} 입니다.")