# for 변수 in collections :
# collections :  list, tuple, set, dict, ...
# iterable -> __iter__, __next__

subjects = ["python", "ds", "de"]
for sub in subjects:
    print(sub)

for i in range(1, 101) :
    print(i, end=", ")
print()

# 중첩
for i in range(1, 10) :
    print(i)
    for j in range(1, 10) :
        print(f"{i} , {j}")


# 구구단 출력해보자
# 2단
# 2 * 1 = 2
# 2 * 2 = 4
# ...
# 9 * 9 = 81
print("구구단")
for i in range(2, 10) :
    print(f"{i}단")
    for j in range(1, 10) :
        print(f"{i} * {j} = {i * j}")

# hello, world를 10번 반복해서 출력
for _ in range(10) :
    print("hello, world!")

# enumerate : (index, value)
print(subjects)
for idx, val in enumerate(subjects) :
    print(f"index: {idx},\t value: {val}")

print(enumerate(subjects))

for i in enumerate(subjects) :
    print(i)