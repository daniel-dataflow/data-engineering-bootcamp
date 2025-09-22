class HelloSelf:

    # 속성(클래스 안에 있는 변수들)
    # class variable : 해당 클래스를 통해 만들어진 모든 객체가 사용 가능
    suffix = "님, 반갑습니다!!"

    # 앞 뒤로 __ 붙은 매서드 :  special method (magic method)
    # --init__(self) (생성자): __new__() 호출하여 객체 생성 /  instance 변수를 초기화
    def __init__(self):
        # 속성
        # instance variable : 객체 각각이 다른 값을 가질 수 있음!
        self.prefix = "안녕하세요, "

    # 기능(함수)
    # function (기능) : 독립적
    # method (기능) : 종속적
    # greetings 는 HelloSelf 라는 class 안에 들어가 있다. =  method
    # self : 나 자신 객체 (instance)
    def greetings(self, name):
        print(f"{self.prefix}  {name} {HelloSelf.suffix}")

if __name__ == "__main__":
    class02 = HelloSelf()
    class02.greetings("한대성")

    # greetings는 self(나 자신 , instance)가 필요하다!!!!
    # = 객체로 만들어져야 사용 가능하다.
    # HelloSelf.greetings("한대성")

    print(HelloSelf.suffix)
    # print(HelloSelf.prefix)
    print(class02.prefix) # 객체가 만들어진 것에서 가져와야 한다.
    print(class02.suffix) 