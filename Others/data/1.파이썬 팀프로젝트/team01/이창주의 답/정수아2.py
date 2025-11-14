import re
v_text = '내 전자 편지의 주소는 example@email.com 이야. 여기로 연락줘.'
v_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+',v_text)
print(v_match.group())