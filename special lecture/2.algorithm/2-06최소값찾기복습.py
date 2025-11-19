## 함수
from random import randint
def findMinIdx(ary):
    minIdx = 0
    for i in range(1, len(ary)):
        if ary[minIdx] > ary[i]:
            minIdx = i

    return minIdx

## 변수
#testAry = [55, 88, 33, 77]
testAry = [randint(10, 299) for _ in range(20)]

## 메인
minPos = findMinIdx(testAry)
print('최솟값 -->', testAry[minPos])
print('정렬 전 -->', testAry)
