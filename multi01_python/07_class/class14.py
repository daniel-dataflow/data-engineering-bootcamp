# 다이어몬드 상속 - 다중상속
class Person:
    def prn(self):
        print("person")

class Father(Person):
    def prn(self):
        print("father")

class Mother(Person):
    def prn(self):
        print("mother")

class Child(Father, Mother): # 상속받은 순서대로 나온다
    pass

if __name__ == "__main__":
    child = Child()
    child.prn()

    # 매서드 탐색 순서 (MRO : Method Resolution Order) 에서 가까운 순서대로
    print(Child.mro()) #[<class '__main__.Child'>, <class '__main__.Father'>, <class '__main__.Mother'>, <class '__main__.Person'>, <class 'object'>]
