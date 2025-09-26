from threading import Thread


class Animal(Thread):
    def __init__(self, name, msg):
        Thread.__init__(self)
        self.name = name
        self.mgs = msg

    def run(self):
        for i in range(3):
            print(f"{self.name} : {self.mgs} ({i}번지)")

if __name__ == "__main__":
    thread01 = Animal("멍멍이1","멍멍!")
    thread02 = Animal("멍멍이2", "멍멍!")
    thread03 = Animal("야옹이1", "야옹!")
    thread04 = Animal("야옹이2", "야옹!")

    thread01.start() #.start() 가 run()을 호출하게 된다.
    thread02.start()
    thread03.start()
    thread04.start()
