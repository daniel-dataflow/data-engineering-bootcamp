# match ~ case
from unittest import case

a = input("1 ~ 3 사이의 정수값을 입력해 주새요.")

match a :
    case "1" :
        print("one")
    case "2" :
        print("two")
    case "3" :
        print("three")

season = input("월 입력 : ")
match int(season) :
    case 12| 1 | 2 :
        print("겨울")
    case 3 | 4 | 5 :
        print("봄")
    case 6 | 7| 8 :
        print("여름")
    case 9 | 10 |11 :
        print("가을")
    case _ :
        print("1 ~ 12 사이의 값만 입력해 주세요!!")

# _ :  특별히 값을 사용하고 싶진 않은데, 변수가 필요할 때... -변수대용으로 사용됨, 사용하고 싶지 않은 값을 담는 용도로

