## 함수
def multiplyNumber(num):
    if num <= 1:
        return 1
    return num * multiplyNumber(num -1)

## 메인
print(multiplyNumber(10))