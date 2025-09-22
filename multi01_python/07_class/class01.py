# 설계도
class HelloClass:

    def greetings(selfself):
        print("Hello World!")




if __name__ == '__main__':
     # 변수 = 클래스() -> 객체화 = 메모리에 적제된다.변수에다가 클래스 주소값을 넣어놨다.
    class01 = HelloClass()
    class01.greetings()

    # instance 저장되어 있는 memory 주소
    print(class01)

    # isinstance(객체, 클래스) : 해당 객체가 클래스의 인스턴스인지 아닌지 판별
    print(isinstance(class01, HelloClass))


"""
특징 (4가지)
- 추상화 (abstraction)
- 상속 (inheritance)
- 다형성 (polymorphism)
- 캡슐화 (encapsulation)


원칙 (5가지)
- SRP (단일 책임 원칙) 
- OCP (개방 패쇄 원칙)
- LSP (리스코프 치환 원칙)
- ISP (인터페이스 분리 원칙)
- DIP (의존 역전 원칙)
"""