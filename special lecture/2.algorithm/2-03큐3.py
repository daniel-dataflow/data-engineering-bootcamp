## 함수

def isQueueFull() :
    global SIZE, queue, front, rear
    # 뒤에 여유 있음
    if rear != SIZE-1 : 
        return False
    # 모두 꽉참
    elif (rear == SIZE -1) and (front == -1) : 
        return True
    # 앞쪽 여유
    else :
        # 문별부터 끝까지
        for i in range(front+1, SIZE) :
            queue[i-1] = queue[i]
            queue[i] = None
        front -= 1
        rear -= 1
        return False
def enQueue(data) :
    global SIZE, queue, front, rear
    if isQueueFull() :
        print('더이상 들어 올 수 없습니다.')
        return
    rear += 1
    queue[rear] = data

def isQueueEmpty() :
    global SIZE, queue, front, rear
    if front == rear :
        return True
    else :
        return False
def deQueue() :
    global SIZE, queue, front, rear
    if isQueueEmpty() :
        print('큐가 비었습니다.')
        return None
    front += 1
    data = queue[front]
    queue[front] = None # 생략가능
    return data
    
def peek() :
    global SIZE, queue, front, rear
    if isQueueEmpty() :
        print('큐가 비었습니다.')
        return None
    return queue[front+1]

## 변수
SIZE = 5
queue = [None for _ in range(SIZE)]
front = rear = -1

## 메인
enQueue('화사')
enQueue('솔라')
enQueue('문별')
enQueue('휘인')
enQueue('선미')

print('출구<--', queue, '<--입구')

returnData = deQueue()
print('손님 이리로 오세요 : ', returnData)
print('다음 손님 준비하세요 : ', peek())
returnData = deQueue()
print('손님 이리로 오세요 : ', returnData)
returnData = deQueue()
print('손님 이리로 오세요 : ', returnData)
print('출구<--', queue, '<--입구')

enQueue('재남')
print('출구<--', queue, '<--입구')