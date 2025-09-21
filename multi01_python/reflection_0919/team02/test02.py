from random import shuffle
# 사다리 타기

# 입력 값을 받는다.
# 동일한 갯수의 출력값을 받는다.
# 서로 랜덤으로 매칭을 시킨다.
# 결과 값을 짝 지어서 출력한다.

def ladder(people_list, choice_list):
    shuffle(people_list)
    shuffle(choice_list)
    print(f"\n \n 두그두그두그 두~~~~~~!!!")
    for k in range(len(people_list)):
        print(f"{people_list[k]}님은, {choice_list[k]}가 매칭되었습니다.")

    return


if __name__ == '__main__':

    people = int(input("응모할 사람 수를 입력하세요."))

    people_list = []
    choice_list = []

    for i in range(1, people + 1):
        name = input(f"{i}번째 이름을 입력해 주세요 : ")
        people_list.append(name)

    for j in range(1, people + 1):
        choice = input(f"{j}번째 항목을 입력해 주세요 : ")
        choice_list.append(choice)

    ladder(people_list, choice_list)
