#4. 입력을 받아서 한 문장 내에서 각 단어가 나오는 빈도를 나타내시오.
v_sentence = input('원하는 한 문장:')
v_word_count = {}
for v_word in v_sentence.split():
   v_word_count[v_word] = v_word_count.get(v_word,0) + 1
print(v_word_count)