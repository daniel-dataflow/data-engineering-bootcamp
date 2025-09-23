from abc import ABCMeta, abstractmethod


# metaclass : class를 만들어주는 class
class AbstractParent(metaclass=ABCMeta):
    def prnt(self):
        print("추상 클래스")

    @abstractmethod
    def abstract_method(self):
        pass

class Child(AbstractParent):
    def abstract_method(self):
        print("추상 매서드")

if __name__ == '__main__':
    child = Child()
    child.prnt()
    child.abstract_method()