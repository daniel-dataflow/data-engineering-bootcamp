## 함수 선언부
def add_data(friend):
    # 1단계 : 빈칸추가
    katok.append(None)
    kLen = len(katok) # 카톡 크기
    # 2단계 : 마지막 칸에 친구 넣기
    katok[kLen-1] = friend

def insert_data(position, friend): # 3, '미나'
    # 1단계 : 빈칸 추가
    katok.append(None)
    kLen = len(katok) # 전체 크기
    # 2단계 : 한칸씩  뒤로 이동, 마지막 친구 ~ 3등
    for i in range(kLen-1, position, -1):
        katok[i] = katok[i-1]
        katok[i-1] = None
    
    # katok[6] = katok[5]
    # katok[5] = None
    # katok[5] = katok[4]
    # katok[4] = None
    # katok[4] = katok[3]
    # katok[3] = None
    # 3단계 : 비워진 자리에 친구 넣기
    katok[position] = friend

def delete_data(position): # 4
    # 1단계 : 4번째 친구 삭제  
    katok[position] = None
    kLen = len(katok)
    # 2단계 : 5등 부터 마지막까지 한칸씩 앞으로
    for i in range(position, kLen-1):
        katok[i] = katok[i+1]
        katok[i+1] = None
        
    # for i in range(position+1, kLen, 1):
    #     katok[i-1] = katok[i]
    #     katok[i] = None

    # katok[4] = katok[5]
    # katok[5] = None
    # katok[5] = katok[6]
    # katok[6] = None
    # 3단계 : 마지막 칸 완전 삭제
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
