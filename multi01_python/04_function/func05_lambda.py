# lambda 표현식 :  익명 함수 표현식
# lambda param: expression

def sum10(x, y):
    return x + y + 10

print(sum10(1, 2))

# 변수에 담아서 사용할거면, 굳이 lambda 쓰지 말고 함수 선언해!
sum20 = lambda x, y: x+y +20 #신택스 슈가
print(sum20(1, 2))
# 함수를 간단하게 한번만 사용하고 싶을 때 lambda 사용!

print((lambda x,y,z: x * y / z)(10, 20, 30))

tuple_list = [
    (1, "one", 9), (2, "two", 8),
    (4, "four", 6),(3, "three", 7)
]
print(tuple_list)

tuple_list.sort()
print(tuple_list)

tuple_list.sort(key=lambda x : x[1])
print(tuple_list)



# 중첩도 가능
result = (lambda x : lambda y: x + y)(10)(20)
print(result)

nested = (lambda x : lambda y : x + y)(10)
print(nested)                                   # lambda y : 10 + y
nested_result = nested(20)
print(nested_result)


# 1 ~ 100 사이의 리스트
hundred = [i for i in range(1,101)]
print(hundred)
# 1 ~ 100 사이에서 5의 배수만 가지는 list
fiv01 = [i for i in range(1,101) if not (lambda x : bool(x % 5))(i)]
print(fiv01)
fiv02 = [i for i in range(1, 101) if not (lambda  x : x if (x %5) != 0 else None)(i)]
print(fiv02)

# zip (두개의 리스트가 같은 길이를 가져야 하는 전재가 있음)
num = [1,2,3]
char = ["a", "b", "c"]
zip_test = zip(num, char)
print(zip_test)
print(list(zip_test))
print(dict(zip(num, char)))

# map : 각각의 값에 함수를 실행
map_test = map(lambda x: x * 10, num)
print(list(map_test))

print(list(map(lambda x: x.upper(), char)))

# filter
print(filter(lambda x: x > 1, num))
print(list(filter(lambda x: x > 1, num)))

# 숙제
# test_list 에서 숫자만 새로운 list로 만들어서 출력하자
test_list = ["3", "6", None, "9", "", "a"]
# hint! str.isdigit()
# hint!! list, filter, lambda
print(list(filter(lambda x: bool(x.isdigit()) , test_list)))
