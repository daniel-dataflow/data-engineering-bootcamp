def coffee(quantity, price, type):

    change = price - (quantity * type * 200)
    if change >= 0 :
        prn(quantity, change)
    else:
        prn()

def prn(quantity=0, change=0):
    if quantity == 0 & change == 0:
        print("금액이 부족합니다.")
    else:
        print(f"커피 {quantity}잔과 잔돈 {change}원이 나왔습니다.")

def start():
    t = int(input("일반커피는 1번을 고급커피는 2번을 눌러주세요."))
    q = int(input("커피 몇 잔이 필요하신가요?"))
    p = int(input("금액을 넣어주세요 (일반 커파는 1잔에 200원, 고급 커피는 400원)"))

    coffee(q, p, t)
