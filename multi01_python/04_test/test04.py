# fibonacci numbers
# 0 1 1 2 3 5 8 13 21 34
def fibo(n):
    fibo_list = []
    f, s = 0, 1
    for i in range(n) :
        fibo_list.append(f)
        temp = f
        f = s
        s = s + temp

    return print(*fibo_list)

"""
def fibo(n):
    a, b = 0, 1
    i = 0
    while i < n :
        print(a, end=" ")
        a, b = b, a + b
        i += 1
"""

if __name__ == '__main__':
    n = int(input('출력할 갯수 입력 : '))
    fibo(n)


# f     0 1 1 2 3 5  8  13 21 34
# temp  0 1 1 2 3 5  8  13 21 34
# s     1 2 3 5 8 13 21 34 55 89
# n     1 2 3 4 5 6  7  8  9  10