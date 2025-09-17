# mutable : list, set, dictionary - 값이 변할 수 있는
a = [1, 2, 3, 4, 5]
print(a)
print(id(a)) # 어디에 저장되어 있는지

a.append(6)
print(a)
print(id(a)) # 같은 메모리에서 수정된다.

# immutable : numbers, tuple, str, frozenset - 값이 변하지 않는
b = 10
print(b)
print(id(b))

b = 20
print(b)
print(id(b))

c = (1, 2, 3, 4, 5)
print(c)
print(id(c))

c = c + tuple(a)
print(c)
print(id(c)) # 새로운 데이터 생성되고 기존의 데이터는 연결 없이 가비지 형태로 남게 된다.

