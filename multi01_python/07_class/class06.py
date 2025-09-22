class HelloPrivate:

    def __init__(self):
        self.public = "public!!!"
        # private : __ 변수 앞에 작성
        self.__private = "private!!!"

    # __ 함수: private
    def __private_method(self):
        print(f"private method : {self.public}, {self.__private}")

    def public_method(self):
        print(f"public method : {self.public}, {self.__private}")
        # class 내부에서는 사용 가능!
        self.__private_method()

if __name__ == '__main__':
    class06 = HelloPrivate()
    print(class06.public)
    class06.public_method()

    # class 내부에서만 사용가능하다.
    # print(class06.__priave)
    # class.__private_method()