"""
a = input()
print(a)
print(type(a))
"""
"""
name = input ("이름 입력 : ")
print(f"안녕하세요, {name} 님!!!")
"""

# 이름, 국어점수, 수학점수, 영어점수를 입력받아서
# dictionary에 keys는 각각 name, kor, math, eng로 저장하고,
# 해당 dictionary에 key는 sum, average로 하여 총 합과 평균을 저장하고
# 해당 dictionary를 출력하자.
name = input("이름 입력 : ")
kor = input("국어점수 : ")
math = input("수학점수 : ")
eng = input("영어점수 : ")

score = dict()
score["name"] = name
score["kor"] = kor
score["math"] = math
score["eng"] = eng
score["sum"] = int(kor) + int(math) + int(eng)
score["average"] =( int(kor) + int(math) + int(eng)) // 3

print(f"안녕하세요 , {score['name']}님! 당신의 총 점수는{score['sum']}이고 평균점수는 {score['average']}입니다.")