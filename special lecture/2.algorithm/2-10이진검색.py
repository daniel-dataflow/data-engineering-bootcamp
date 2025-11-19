## 함수
from random import randint, choice
def binSearch(ary, fData):
    pos = -1
    # 멍멍이 위치
    start = 0
    # 아빠 위치
    end = len(ary)-1
    # 시작이 끝보다 작거나 같을때 까지
    while (start <= end):
        # 중앙 위치
        mid = (start + end) // 2
        # 찾았어?
        if ary[mid] == fData:
            pos = mid
            break
        # 시작을 중앙의 오른쪽으로 이동(왼쪽 버려)
        elif ary[mid] < fData:
            start = mid + 1
        # 끝을 중앙의 왼쪽으로 이동(오른쪽 버려)
        else :
            end = mid - 1
    return pos


## 변수
dataArray = [randint(50, 190) for _ in range(10)] # 가족 10명
findData = choice(dataArray) # 누나 키
dataArray.sort()

## 메인
print('데이터 -->', dataArray)
position = binSearch(dataArray, findData)
if position == -1 :
    print(findData, '(이)가 없네요.')
else :
    print(findData, '(은)는 ', position, '번째 있어요.')