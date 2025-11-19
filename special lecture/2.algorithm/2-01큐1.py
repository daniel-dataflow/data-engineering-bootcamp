## 함수


## 변수
SIZE = 5
queue = [None for _ in range(SIZE)]
front = rear = -1


## 메인
# enQueue()
rear += 1
queue[rear] = '화사'
rear += 1
queue[rear] = '솔라'
rear += 1
queue[rear] = '문별'

print('출구<--', queue, '<--입구')
# deQueue()
front += 1
data = queue[front]
queue[front] = None # 생략가능
print('추출한 데이터 -->', data)

front += 1
data = queue[front]
queue[front] = None # 생략가능
print('추출한 데이터 -->', data)
print('출구<--', queue, '<--입구')