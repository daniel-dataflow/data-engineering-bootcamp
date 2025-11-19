## 함수
class Node:
    def __init__(self):
        self.data = None
        self.link = None

## 변수


## 메인
node1 = Node() # 빈 노드 생성
node1.data = '다현'

node2 = Node()
node2.data = '정연'
node1.link = node2 # 연결

node3 = Node()
node3.data = '쯔위'
node2.link = node3 # 연결

node4 = Node()
node4.data = '사나'
node3.link = node4 # 연결

node5 = Node()
node5.data = '지효'
node4.link = node5 # 연결

## 노드의 삽입
# 1단계
newNode = Node()
newNode.data = '재남'
# 2단계
newNode.link = node2.link
node2.link = newNode

## 노드의 삭제
# 1단계
node2.link = node3.link
# 2단계
del(node3)


# print(node1.data, end=" ")
# print(node2.data, end=" ")
# print(node3.data, end=" ")
# print(node4.data, end=" ")
# print(node5.data)

head = node1
# print(head.data, end=" ")
# print(head.link.data, end=" ")
# print(head.link.link.data, end=" ")
# print(head.link.link.link.data, end=" ")
# print(head.link.link.link.link.data)

current = head
print(current.data, end=" ")
while current.link != None: # 현재 노드가 링크가 안 비어있을 때
    current = current.link
    print(current.data, end=" ")
print()
