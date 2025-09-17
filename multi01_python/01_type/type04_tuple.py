# tuple :  list와 거의 같음 (변경이 불가능)

# ( 값, 값, 값, ...)
a = (1, 2, 3, 4, 3)
print(a)
print(type(a))

# 생성자
b =  tuple([5,6,7,8])
print(b)
print(type(b))

# a.append(6)
# a.pop(6)
print(dir(a)) # dir : 그 객체가 가지고 있는 속성이나 메소드을 리스트로 출력해준다.

print(a.count(6))

# tuple + tuple
c  = a+b
print(c)
print(type(c))

#형 변환
# tuple -> list
d = list(c)
print(d)
print(type(d))
d.remove(7)
print(d)

# list -> tuple
e = tuple(d)
print(e)
print(type(e))



# packing
f = 1, 2, 3
print(f)

# unpacking
g, h, i = f
print(g)
print(h)
print(i)

"""
j, k = f
print(j)
print(k)
"""
# 거의 대부분 정상종료는 "종료 코드 0(으)로 완료된 프로세스"으로 나타난다.