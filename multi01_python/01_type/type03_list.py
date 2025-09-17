# 여러개의 값을 효과적으로 관리하기 위한 객체들
# -> list, tuple, set, dictionary

# Sequence :  순서가 있는 값들을 가진 객체

# list :  순서 o , 중복 o

# [ 값, 값, 값]
a = [5,3,4,2,3,1,3]
print(a)
print(type(a))

a.append(6)
print(a)

# list() 생성자
b =  list()
print(type(b))
b.append(7)
b.append(9)
b.append(8)
b.append(9)
print(b)


# list에서 특정 값 가져오기
# list[index]
# index는 보통 0부터 시작!
# a = [5,3,4,2,3,1,3,6]
print(a[2])
# b = [7,9,8,9]
print(b[1])


# dir : 객체의 속성, 매서드 확인
print(dir(list))
# __??__ :  special method (magic method)
# __iter__ : 반복가능한 (iterable) =?> iterator
# __len__ , __getitem__ : 연속적인 (squenceable)

# list를 거꾸로 출력
b.reverse()
print(b)

#list의 값을 빼내오고 싶어
# 맨 뒤에서 하나 빼와
print(a)
print(a.pop())
print(a)

#list 정렬하고 싶어
print(a.sort())
print(a)

#list 크기 알려줘
print(len(a))
print(a)

# list 연산
# list + list = ?
c = a + b
print(c)
print(type(c))

# * : 곱하기
print(a * 2)

# index 3 위치에 숫자 999를 넣어보자
c.insert(3,999)
print(c)

# 중첩
d = ['a','b','c','d','e',['f','g','h'], 'i']
print(d)
print(len(d))

# g 출력하고 싶어
print(d[5][1])


