# 산술연산
a = 17
b = 3

print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 거듭제곱 (power)
print(a ** b)

# 몫 (floor division : 소수점 이하는 버림)
print(a // b)

# 나머지 (modulo)
print(a % b)

# 비교연산
a, b = 5, 3
print(a == b)
print(a != b)
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
# print(a =< b)
print(a is b)
print(a is not b)

# 여러개 한번에 비교도 가능
print(1 < b < a )

# and : 둘 다 True 이어야지만 Ture
print(True & True)
print(True and True)

# or : 둘 중 하나만 True 이면 True
print(True | False)
print(True or False)

# not  :  True -> False , False -> True
print(not True)

# 멈베연산
list01 = [1, 2, 3, 4, 5]

print(3 in list01)
print(6 in list01)
print(3 not in list01)

# range  : 해당 범위의 숫자 생성
print(range(10))
# range(stop) : 0 ~  stop -1
print(list(range(10)))
# range(start, stop) :  start ~ stop -1
list02 = list(range(10, 20))
print(list02)
# range(start, stop, step) : start, start + step, start + step + step, ...., stop -1
print(list(range(1, 11, 2)))
# 10 ~ 1 거꾸로
print(list(range(10, 0, -1)))

# slice
original = list(range(100))
print(original)

# [n] : n index
print(original[5])
# [start, stop] -> start index ~ stop index -1
slice01 = original[1:5]
print(slice01)
#[start, stop, step]
slice02 = original[10:20:2]
print(slice02)

# 숫자가 없으면?
print(original[:50])
print(original[50:])
print(original[::10])

# 거꾸로
slice03 = original[20:10:-1]
print(slice03)

hello = "hello, world!"
print(hello)

# !를 빼고 출력하고 싶어요
print(hello[0:12])
print(hello[:12])
print(hello[:-1])

# -1
print(hello[-1])
print(hello[:-1])
print(hello[::-1])

# world만 출력하고 싶어...
print(hello[7:-1])

# 증감연산자
c = 10

# c 라는 변수에 저장된 값에다가
# #1을 증가시켜서
# 다시 c에 저장하고 싶어...
c = c + 1
print(c)
c += 1  #신텍스 슈거
print(c)

c -= 1
print(c)

c *= 2
print(c)

c /= 2
print(c)




