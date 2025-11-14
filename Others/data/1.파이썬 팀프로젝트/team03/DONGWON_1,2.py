# 문1] 1부터 100까지의 숫자 중에서 3의 배수이면서 5의 배수인 수의 중복된 숫자(예: 15, 30 등 3과 5의 공배수)를 제외한 합을 구하는 코드를 작성

# 문2] 입력한 숫자와 랜덤한 숫자(10)가 일치하면 승, 15 입력하면 종료










r






# [문제 1 예시 답안]
# sum_thr = 0
# sum_fiv = 0
# sum_total = 0
#
# for thr in range(1, 101):
#     if thr % 3 == 0:
#         sum_thr += thr
#     elif thr % 5 == 0:
#         sum_fiv += thr
#
#
# sum_total = sum_thr + sum_fiv
#
# print(sum_total)


# [문제 2 예시 답안]
# from random import randint
#
# computer_num = randint(1, 10)
#
# def game_process(player_num: int) -> None:
#
#
#     computer_num = randint(1, 10)
#     print(f"player : {player_num} , computer : {computer_num}")
#
#     if player_num == computer_num :
#         print("승")
#     else:
#         print("다시")
#
# def play() -> None:
#     while True:
#         player_num = int(input('1~10 사이의 숫자를 입력하세요 \n종료는 15 입력 : '))
#
#
#         if player_num == 15:
#             print("종료")
#             break
#
#         elif player_num >= 11:
#             print("1~10 중 입력")
#             continue
#         else:
#             game_process(player_num)
#
#
#
#
# if __name__ == '__main__':
#     play()
#



