# if 조건 :
a = 1

if a == 1 :
    print(f"{a} 는 1 입니다.")

# if ~ else
b = 2
if a ==b :
    print(f"{a} == {b}")
else :
    print(f"{a} != {b}")

# if ~ elif
if a > b :
    print(f"{a} > {b}")
elif a < b :
    print(f"{a} < {b}")
else :
    print(f"{a} == {b}")

wheels = 2
engine = True

if engine :
    if wheels == 2 :
        print("오토바이")
    elif wheels == 4 :
        print("자동차")
else :
    if wheels == 2:
        print("자전거")
