class HelloStaticDeco:

    class_val = "class variable"

    def  __init__(self): # __init__에서 self를 선언함으로써 객체를 생성하고 초기화하는 함수
        self.name = "instance"

    # decorator : 기능을 추가하는 녀석
    # method를 함수처럼 사용
    @staticmethod # 전역으로 바로 사용하고 싶을때
    def static_method(): # self가 없으면 그 클래스가 가지고 있는 함수를 사용할 수 있다.
        print(f"static method :  {HelloStaticDeco.class_val}")


    def instance_method(self): # self가 있으면 나 자신이 객체이기 때문에 사용할려면 변수에 담아서 사용해야만 한다.
        print(f"instance method :  {HelloStaticDeco.class_val}")
        print(self.name)

if __name__ == "__main__":
    class07 = HelloStaticDeco() #물품을 만들어서 사용하겠다.
    class07.static_method()
    class07.instance_method()

    HelloStaticDeco.static_method()
    # self 라고 해서 나 객체가 필요하다 인스턴스가 있어야지만 동작하는 것을
    # HelloStaticDeco.instance_method()