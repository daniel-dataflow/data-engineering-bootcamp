from random import randint


'''
가위바위보 게임
출력 예)

가위 : 1 | 바위 : 2 | 보 : 3 | 게임종료 : 4 -> 숫자를 입력하세요 : 1
player (가위) vs computer(보) : 당신이 이겼습니다.
가위 : 1 | 바위 : 2 | 보 : 3 | 게임종료 : 4 -> 숫자를 입력하세요 : 2
player (바위) vs computer(가위) : 당신이 이겼습니다.
가위 : 1 | 바위 : 2 | 보 : 3 | 게임종료 : 4 -> 숫자를 입력하세요 : 3
player (보) vs computer(가위) : 컴퓨터가 이겼습니다.
가위 : 1 | 바위 : 2 | 보 : 3 | 게임종료 : 4 -> 숫자를 입력하세요 : 4
다음에 또 놀러오세요
'''

class GawiBawiBoinputException(Exception):
    def __init__(self):
        print("1 2 3 4 중 하나만 입력해 주세요 !!!")


# 가위바위보 만들기
def game_process(player_num: int) -> None:
    """
    player - computer
    player              가위 1    바위 2    보 3
    computer    가위 1    0       1       2
                바위 2    -1      0       1
                보  3    -2      -1      0
    승 : -2, 1
    무 : 0
    패 : -1, 2
    """

    prn ={1:"가위", 2:"바위", 3:"보"}

    computer_num = randint(1, 3)
    print(f"player :  {prn[computer_num]} vs computer: {prn[computer_num]}", end= " : ")

    if (player_num - computer_num) in [-2, 1]:
        print("당신이 이겼습니다!!!")
    elif (player_num == computer_num):
        print("비겼습니다...")
    else:
        print("컴퓨터가 이겼습니다.ㅠㅠ")


def play() -> None:
    while True: # 무한반복
        try:
            player_num = int(input("가위 : 1 | 바위 : 2 | 보 : 3 | 게임종료 : 4 -> 숫자를 입력하세요 : "))

            if player_num not in [1, 2, 3, 4]:
                raise GawiBawiBoinputException()
            if player_num == 4:
                break
        except ValueError:
            print("숫자만 입력하실 수 있습니다...")
            print("다시 입력해 주세요.")
        except GawiBawiBoinputException:
            # print("1, 2, 3, 4 중에 하나만 입력하실 수 있습니다...")
            print("다시 입력해 주세요.")
        except: # except Exception 이랑 같은 의미
            print("문제가 생겼습니다. 관리자에게 연락해 주세요...")
            break
        else:
            game_process(player_num)

    print("다음에 또 놀러오세요.")



if __name__ == '__main__':
    play()



