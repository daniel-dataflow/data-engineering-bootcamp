def exception01():
    print(10 / 0)

def exception02():
    try:
        print(10 / 0)
    except:
        print("0으로 나눌 수 없습니다!!!")

def exception03():
    try:
        print(10 / 0)
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다!!!!")

def exception04():
    try:
        a =[1, 2, 3, 4, 5]
        print(a[5])
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다!!!")

def exception05():
    try:
        a =[1, 2, 3, 4, 5]
        print(a[5])
    except ZeroDivisionError:
        print("0으로 나눌 수 없습니다!!!")

    except IndexError:
        print("index를 다시 확인해 주세요")

def exception06():
    try:
        a =[1, 2, 3, 4, 5]
        print(a[5])
    except ZeroDivisionError as e:
        print(e)

    except IndexError as e:
        print(e)

def exception07():
    try:
        a =[1, 2, 3, 4, 5]
        print(a[5])
    except Exception as e:
        print(e)

def exception08():
    try:
        a =[1, 2, 3, 4, 5]
        print(a[5])
    except Exception as e:
        print("예외 발생!!!")
    else:
        print("예외 발생 안함...")
    finally:
        print("무조건 실행!!!!!")

def exception09():
    try:
        a =[1, 2, 3, 4, 5]
        print(a[4])
    except Exception as e:
        print("예외 발생!!!")
    else:
        print("예외 발생 안함...")
    finally:
        print("무조건 실행!!!!!")

if __name__ == '__main__':
    # exception01() # 에러나는 코드
    # exception02() # 모든 에러를 예외처리 해준다.
    # exception03() # 예상가능한 에러가 있을 경우 그 에러 코드에 맞게 작성할 수도 있다.
    # exception04() # 잘못된 익셉션 예제
    # exception05() # 여러개가 날 경우 여러개를 예측해서 나타낼 수 있다
    # exception06() # 에러메시지를 직접 출력할 수 있다.
    # exception07() # 잘 모르는 에러를 출력 해줄 수 있게
    # exception08() #예외 발생 파아널리 출력
    exception09() #try, else, finally 출력됨 -  finally 에서는 close를 많이 함