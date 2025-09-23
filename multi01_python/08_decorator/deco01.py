def outer():

    prefix = "outer"

    def inner():
        print(f"{prefix} inner")

    # closure : lexical environment 기억!
    return inner

# 함수가 값으로 들어올거야. (일급객체 or 일급함수)
# 2)
def func_param(func):
    # 3)
    # 4)
    func()()

if __name__ == "__main__":
    # 1)
    func_param(outer)
