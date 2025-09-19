# 휴게소에서 음식을 주문하는 키오스크를 만든다.
# 음식점을 고른다.
# 음식점에서 음식을 고른다.
# 수량을 누른다.
# 금액을 넣는다.

def calculate(quantity, price, type):

    change = price - (quantity * type * 1000)
    if change >= 0 :
        prn(quantity, change)
    else:
        prn()

def prn(quantity=0, change=0):
    if quantity == 0 & change == 0:
        print("금액이 부족합니다.")
    else:
        print(f" {quantity}잔과 잔돈 {change}원이 나왔습니다.")

def start():
    r = int(input("원하시는 식당을 골라주세요. 한식당은 1번을 중식당는 2번을 눌러주세요."))
    if r == 1:
        t = korean_restaurant()
    elif r == 2:
        t = chinese_restaurant()
    else:
        print("없는 식당입니다. 시스템을 종료합니다. 다시 시작해주세요.")
    q = int(input("수랴을 선택 해주세요."))
    p = int(input("금액을 넣어주세요."))
    calculate(q, p, t)

def korean_restaurant():
    t = int(input("원하시는 메뉴을 골라주세요. 1번 육계장 10,000원, 2번 곰탕 12,000원 3번 순두부찌개 8,000원"))
    return t

def chinese_restaurant():
    t = int(input("원하시는 메뉴을 골라주세요. 1번 짜장면 8,000원, 2번 짬뽕 10,000원 3번 탕수육 20,000원"))
    return t



if __name__ == '__main__':
    start()