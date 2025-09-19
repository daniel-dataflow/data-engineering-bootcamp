# 해석하는 숙제
# 1

"""
1) ['*' for i in range(i + 1)] : i가 0부터 i+1까지 반복하면서 * 생성 -> list에 담아라
=> ["*"],
=> ["*", "*"]
...
2)  [''.join(['*' for i in range(i + 1)]) for i in range(5)] : join은 리스트를 문자열로 바꿔준다.
=> ["*"] => "*"
=> ["*", "*"] => "**"
...
=>["*", "**", "***", ...] => "*****"
3) '\n'.join([''.join(['*' for i in range(i + 1)]) for i in range(5)])
=> "*\n**\n***\n****\n*****"
print("*\n**\n***\n****\n*****")
"""
print('\n'.join([''.join(['*' for i in range(i + 1)]) for i in range(5)]))
print('-----')

# 2
print('\n'.join([''.join(['*' for i in range(i)]) for i in range(5, 0, -1)]))
print('-----')

# 3
print('\n'.join([''.join([' ' for i in range(4 - i)] + ['*' for i in range(i + 1)]) for i in range(5)]))
print('-----')

# 4
print('\n'.join([''.join([' ' for i in range(i)] + ['*' for i in range(5 - i)]) for i in range(5)]))
print('-----')

# 5
print('\n'.join([''.join([' ' for i in range(4 - i)] + ['*' for i in range(2 * i + 1)]) for i in range(5)]))
print('-----')
