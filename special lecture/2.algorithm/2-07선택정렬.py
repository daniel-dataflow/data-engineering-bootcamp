## 함수
from random import randint
def findMinIdx(ary):
    minIdx = 0
    for i in range(1, len(ary), 1):
        # 가장 작은값과 그 다음 숫자 비교
        if ary[minIdx] > ary[i]:
            minIdx = i
    return minIdx

## 변수
before = [randint(50, 190) for _ in range(8)]
after = []

## 메인
print('정렬 전 -->', before)
# 전체 가족 수만큼 반복
for i in range(len(before)):
    minPos = findMinIdx(before)
    after.append(before[minPos])
    del(before[minPos])

print('정렬 후 -->', after)