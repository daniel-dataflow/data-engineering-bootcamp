## 함수
from random import randint
def selectionSort(ary):
    n = len(ary)
    for i in range(n-1):
        minIdx = i
        for k in range(i+1, n-1):
            if ary[minIdx] > ary[k]:
                minIdx = k
        ary[i], ary[minIdx] = ary[minIdx], ary[i]
    return ary


## 변수
dataArray = [randint(50, 190) for _ in range(8)]

## 메인
print('정렬 전 -->', dataArray)
dataArray = selectionSort(dataArray)
print('정렬 후 -->', dataArray)