def my_generator(end):
    i = 0                       # while 인 경우 스코프가 밖에 변수를 사용할 수 있게 된다.
    while i < end:
        i += 1

        yield i ** 2            # yield: 양보하다  __next__ 가 요청될때마다 해당하는 값을 전달해준다.
                                # yield 가 명시되어 있으면 generator 이다.

if __name__ == "__main__":
    nums = my_generator(10)
    print(nums)
    print(type(nums))

    print(nums.__next__())
    print(nums.__next__())
    print(nums.__next__())