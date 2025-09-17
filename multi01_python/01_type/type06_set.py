# set : 순서 x, 중복 x

a = {"1", "2", "4", "3", "3"}
print(a)

# 숫자는 정렬되더라...
# 숫자는 hashing 할 때 숫자 자체가 주소 값으로 저장됨
b = { 1, 5, 4, 3, 2, 6}
print(b)
print([hash(x) for x in [1, 2, 3, 4, 5]])


# set()
c = set([1, 2, 3, 4, 5])
print(c)

print(c.add(6))
print(c)
print(c.pop())
print(c)

# set() 안에 iterable 한 객체를 넣으면 ...?
d = set("hello, world!")
print(d)

left  = {"a", "b", "c", "d"}
right = {"c", "d", "e", "f"}
print(left.union(right)) # 합집합
print(left | right) #결과가 같은 이유는 같은 메모리에서 가져온 값이기 때문에

print(left.intersection(right)) # 교집합
print(left & right)

print(left.difference(right)) # 차집합
print(left - right)

# frozenset
e = frozenset({1, 2, 3, 4, 5})
print(e)
# e.pop()
print(dir(set))
print(dir(frozenset))
