class Calc:
    def __init__(self, initial_value):
        self.value = initial_value

    def add(self, num):
        self.value += num
        return self

    def subtract(self, num):
        self.value -= num
        return self

    def multiply(self, num):
        self.value *= num
        return self

    def divide(self, num):
        if num == 0:
            print("Error: Cannot divide by zero.")
            return self
        self.value /= num
        return self

    def get_value(self):
        return self.value


if __name__ == "__main__":
    result = Calc(1).add(2).multiply(3).subtract(4).get_value()
    print(f"최종 결과: {result}")
