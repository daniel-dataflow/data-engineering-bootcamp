from abc import ABC, abstractmethod

# ABC : Abstract Base Class :추상 메서드를 가지고 있는 추상 클래스이다.
class AbstractParent(ABC):
    def prn(self):
        print("abstract class :  abstract method를 가질 수 있는 class")

    @abstractmethod # 자식클래스들 중에서 반드시 구현해야 한다.: 추상클래스를 만들어주는 실질적인 역할을 한다.
    def abstract_method(self):
        pass

class Child(AbstractParent):
    def abstract_method(self):
        print("abstract method : 상속받은 자식 클래스에서 반드시 구현!")

if __name__ == "__main__":
    # abstract_parent = AbstractParent()
    # abstract_parent.prn()
    # abstract_parent.abstract_method()
    # 추상 클래스 (AbstractParent)는 객체생성 불가!

    child = Child()
    child.prn()
    child.abstract_method()