## 함수 선언부
def add_data(friend):
    katok.append(None)
    kLen = len(katok) # 카톡 크기
    katok[kLen-1] = friend

def insert_data(position, friend): # 3, '미나'
    katok.append(None)
    kLen = len(katok) # 전체 크기
    for i in range(kLen-1, position, -1):
        katok[i] = katok[i-1]
        katok[i-1] = None
        katok[position] = friend
    

def delete_data(position): # 4
    katok[position] = None
    kLen = len(katok)
    for i in range(position, kLen-1, 1):
        katok[i] = katok[i+1]
        katok[i+1] = None
    del(katok[kLen-1])

    

## 전역 변수부
# katok = ['다현','정연','쯔위','사나','지효']
katok = []

## 메인 코드부
# 데이터 삽입 : 모모에게 카톡 1회.
add_data('다현')
add_data('정연')
add_data('쯔위')
add_data('사나')
add_data('지효')
print(katok)
add_data('모모')
print(katok)
insert_data(3, '미나')
print(katok)
delete_data(4)
print(katok)
