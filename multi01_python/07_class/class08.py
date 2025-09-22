class HelloClassDeco:

    # class 변수 : 해당 class로 만들어진 instance 전체가 공유 = static
    cnt = 0

    def __init__(self): # self는 -객체
        self.name = "class decorator"
        HelloClassDeco.cnt += 1

    # classmethod decorator :  class를 받는다.
    # staticmethod decorator :  다른점은 상속 일때...
    @classmethod
    def class_method(cls):
        # cls : class 별침 - 설계도
        print(cls.cnt)

    # 객체가 메모리에서 반환될 때 호출
    # garbage collected
    def __del__(self):
        HelloClassDeco.cnt -= 1





if __name__ == '__main__':
    class08_1 = HelloClassDeco() # 이런 형태는 __init__을 호출하고 있어.
    class08_2 = HelloClassDeco()
    class08_3 = HelloClassDeco()

    print(class08_1.cnt)
    class08_2.class_method()
    HelloClassDeco.class_method()


    del class08_3 #class08_3가 삭제되는 순간 __del__를 실행한다. 메모리에서 class08_3를 삭제하였다.
    # print(class08_3)
    HelloClassDeco.class_method()