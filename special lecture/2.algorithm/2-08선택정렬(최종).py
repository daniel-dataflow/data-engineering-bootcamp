## 함수
from random import randint
def selectionSort(ary):
    # 데이터 개수
    n = len(ary)
    # 사이클 (큰 회전)
    for i in range(0,n-1):
        minIdx = i
        # 사이클 (작은 회전)
        for k in range(i+1, n):
            # 키 비교
            if ary[minIdx] > ary[k]:
                minIdx = k
        # 차이가 가장 작은 데이터와 그 다음 데이터를 고객
        ary[i], ary[minIdx] = ary[minIdx], ary[i]
    return ary


## 변수
dataArray = [randint(50, 190) for _ in range(8)]

## 메인
print('정렬 전 -->', dataArray)
dataArray = selectionSort(dataArray)
print('정렬 후 -->', dataArray)