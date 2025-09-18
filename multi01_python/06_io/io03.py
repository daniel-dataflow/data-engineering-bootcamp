msg = input("파일에 추가적으로 입력할 내용 : ")

# utf-8 : ???
# windows os : cp949 (encoding)

file = open("test01.txt", "a", encoding="utf-8")

file.write(msg + "\n")

file.close()