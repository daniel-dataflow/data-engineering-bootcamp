class HelloGetterSetter:

    def __init__(self, name ="홍길동"):
        self.name = name
        # 변수 앞에 __ : private -> class 내부에서만 사용 가능!
        # 왜 굳이! 변수를 privete 하게 만들고, getter, setter를 만들까?
        self.__age = 100

    # @ ~ : decorator : 기능 추가
    # getter
    @property
    def age(self):
        return self.__age

    # setter
    @age.setter
    def age(self, age):
        self.__age = age

    def __str__(self):
        return f"{self.name}님은 {self.__age}세 입니다."

if __name__ == "__main__":

    class05 = HelloGetterSetter("김선달")
    print(class05)

    # privete : class외부에서 사용불가
    # print(class05.__age)

    class05.age = 20 # 변수가 아니라 메소드다.
    print(class05)
    print(class05.age)