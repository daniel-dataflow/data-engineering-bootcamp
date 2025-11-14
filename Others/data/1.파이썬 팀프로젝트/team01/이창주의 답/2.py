#2. 입력을 받아서 중복 문자를 제거하시오.(순서는 유지)
v_string = input('중복으로 alphabet가 있는 영단어:')
v_result = ''
for v_char in v_string:
   if v_char not in v_result:
      v_result += v_char
print(v_result)