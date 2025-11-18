## 함수
class Node:
    def __init__(self):
        self.data = None
        self.link = None
def printNodes(start): # start 노드부터 끝까지 출력
    current = start
    print(current.data, end=' ')
    while current.link != None:
        current = current.link
        print(current.data, end=' ')
    print()

def insertNode(findData, insertData):
    global memory, head, pre, current
    # Case1 : 하필 머리 앞에 삽입. 다현, 화사
    if findData == head.data:
        # 1단계
        node = Node()
        node.data = insertData
        # 2단계
        node.link = head
        head = node
        memory.append(node) # 생략 가능!
        return  
    # Case2 : 중간 노드 앞에 삽입. 사나, 솔라
    current = head
    while current.link != None: # 마지막까지 찾기...
        pre = current
        current = current.link
        if current.data == findData:
            # 1단계
            node = Node()
            node.data = insertData
            # 2단계
            node.link = current
            pre.link = node
            memory.append(node) # 생략 가능!
            return
    # Case3 : 찾는 노드가 없을 때 . 재남, 문별
    node = Node()
    node.data = insertData
    current.link = node
    memory.append(node) # 생략 가능!
    return
    
def deleteNode(deleteData):
    global memory, head, pre, current
    # Case1 : 하필 삭제할 데이터가 헤드일 때!!!
    if deleteData == head.data:
        # 그림1
        current = head
        # 그림2
        head = head.link
        # 그림3
        del(current)
        return
    # Case2 : 중간 or 마지막 노드 삭제
    current = head
    while current.link != None:
        pre = current
        current = current.link
        if current.data == deleteData:
            # 그림4
            pre.link = current.link
            # 그림5
            del(current)
            return
    # Case3 : 삭제할 데이터가 없을 때
    return

def findNode(findData):
    global memory, head, pre, current
    current = head
    if current.data == findData:
        # 노드(음악, 무비,...)를 통째로 리턴
        return current
    while current.link != None:
        current = current.link
        if current.data == findData:
            # 노드(음악, 무비,...)를 통째로 리턴
            return current
    return None()

## 변수
memory = []
head, current, pre = None, None, None
dataArray = ['다현','정연','쯔위','사나','지효'] # 여러분 데이터



## 메인
# 데이터 배열을 가지고 연결리스트 생성 (핵심)
# 첫번째 노드
node = Node() # 첫번째 노드
node.data = dataArray[0]
head = node
memory.append(node) # 생략 가능!
# 두번째 이후 노드 (동일)
for data in dataArray[1:]: # '정연' ~ '지효'
    pre = node # 이전 노드 기억
    node = Node() # 새 노드 생성
    node.data = data
    pre.link = node # 이전 노드에 새 노드 연결
    memory.append(node) # 생략 가능!
printNodes(head)

# insertNode('다현', '화사') # 다현을 찾아서 그 앞에 회사 삽입 Case1
# insertNode('사나', '솔라') # 사나를 찾아서 그 앞에 솔라 삽입 Case2
# insertNode('재남', '문별') # 재남을 찾아서 그 앞에 문별 삽입 Case3

# deleteNode('다현') # 다현을 찾아서 삭제 Case1
# deleteNode('쯔위') # 쯔위를 찾아서 삭제 Case2

printNodes(head)
# print("전체 노드 수 : ", len(memory))
# print("메모리 주소 : ", memory)

fNode = findNode('사나') # 사나의 블록(음악, 뮤비, ...) 이 통쩨로 리턴
print(fNode.data, '뮤비 플레이!! 쿵짝쿵짝...')