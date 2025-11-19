## 함수
from random import randint, choice
def binSearch(ary, fData):
    pos = -1
    start =0
    end = len(ary) - 1
    while(start <=end):
        # 중앙 찾기
        mid = (start +  end) //2
        if ary[mid] == fData:
            pos = mid 
            break
        elif ary[mid] < fData:
            start = mid + 1
        else:
            end = mid -1
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