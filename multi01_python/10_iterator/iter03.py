class MyIter:
    def __init__(self, end):
        self.start = 0
        self.end = end
        self.val = 1

    def __iter__(self):
        return self

    def __next__(self):
        self.start += 1
        if self.start <= self.end:
            self.val = self.start ** 2
            return self.val
        else:
            raise StopIteration # 예외 부모클래스가 Exception

    def __getitem__(self, item): # 대괄호로 index를 가져오고 싶을때
        if self.start <= self.end:
            return list(self)[item]
        else:
            raise IndexError


if __name__ == "__main__":
    for i in MyIter(10):
        print(i)

    print(list(MyIter(5)))

    print(MyIter(5)[1])  # == print(MyIter(5).__getitem__(1))