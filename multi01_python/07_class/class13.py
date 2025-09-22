class Lion:
    def bark(self):
        print("어흥")

class Eagle:
    def fly(self):
        print("파닥파닥")

# 다중 상속
class Gruffin(Lion, Eagle):
    pass

if __name__ == "__main__":
    griffin = Gruffin()
    griffin.bark()
    griffin.fly()


