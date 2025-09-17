from itertools import count

list01 = list()
for i in range(1, 11):
    list01.append(f"a{i}")

print(list01)

list02 = [f"a{i}" for i in range(1, 11)]
print(list02)

# 1~ 10 사이의 짝수만 list로
list03 = [i for i in range(1, 11) if i % 2 == 0]
print(list03)

subject = [
    "python", "analysis", "database",
    "html", "css" ," django",
    "science", "engineering"
]
# subjects 안의 item들 중에 a를 포함하고 있는 item들만 새로운 list로 만들자!
list04 = [sub for sub in subject if 'a' in sub]
print(list04)

# 중첩
# [ [0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
list05 = [[4 * i + j for j in range(4)] for i in range(4)]
print(list05)