class Parent():

    def __init__(self):
        self.name = "Parent"
        self.age = 100

    def prn01(self):
        print(f"prn01 : {self.name}")

    def prn02(self):
        print(f"prn02 : {self.name} / {self.age}")

class Child(Parent):

    def __init__(self):
        super().__init__()
        self.name = "Child"

    # 부모의 prn01과 동일한 모습
    # override (재정의)
    def prn01(self):
        print(f"override : 부모의 것들을 가지고와서 내가 필요한 모습으로 재정의")

    def prn02(self):
        super().prn02()
        print(f"당연히, 부모의 것들을 호출도 가능해요!")

if __name__ == "__main__":
    child = Child()
    child.prn01()
    child.prn02()