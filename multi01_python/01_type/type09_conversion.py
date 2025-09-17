# 형 변환
from dis import print_instructions

a = int(1.2)
print(a)
b =  int("12")
print(b)
# c = int("12.3") 실수 형태는 안된다.
# print(c)

print(int('1111', 2))
print(int('77', 8))
#print(int(77, 80)) 문자열로 되어 있는 숫자만 형변환 가능

print(int(True))
print(int(False))

print(float("12.5"))
