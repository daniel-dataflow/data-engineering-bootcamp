# print
# flush  :  stream에 남아있는 데이터를 강제로 출력

name = "DaeSung"
age = 30

# name과 age를 출력하자
# print(name + age) 스트링은 넘버와 같이 출력 안된다 타입에러
print(name + str(age))
print(name, age)
# *args /  **kwargs
print(name, age, sep="-")
print(name, age, end="출력이 끝난 후 end가 출력됩니다...")
print("??")
print("name", name, sep=":", end="\t")
print("age", age, sep=":")

# % values
print("name: %s" % name)
print("name : %s \t age: %d" % (name, age))

# str.format()
print("name : {} \t age: {}".format(name, age))

# f-string
print(f"name : {name} \t age: {age}")