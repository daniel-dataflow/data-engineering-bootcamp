my_iter = iter([1, 2, 3, 4, 5]) # == [1, 2, 3, 4, 5].__iter__()
print(my_iter)
print(type(my_iter))

print(next(my_iter))
print(next(my_iter))
print(next(my_iter))
print("-"*10)

# for문은 iterable한 객체 (colle
for i in my_iter:
    print(i)