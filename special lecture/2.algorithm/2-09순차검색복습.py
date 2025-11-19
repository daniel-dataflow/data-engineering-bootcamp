## 함수
from random import randint, choice
def seqSearch(ary, fData):
    pos =-1
    for i in range(len(ary)-1):
        if ary[i] == findData:
            pos = i
            break
    return pos


## 변수
dataArray = [randint(50, 190) for _ in range(8)] # 가족 8명
findData = choice(dataArray) # 누나 키

## 메인
print('데이터 -->', dataArray)
position = seqSearch(dataArray, findData)
if position == -1 :
    print(findData, '(이)가 없네요.')
else :
    print(findData, '(은)는 ', position, '번째 있어요.')