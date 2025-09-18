file = open("test01.txt", "r", encoding="utf-8")

# 전체 다 나옴
# read_txt = file.read()
# print(read_txt)

# 한줄만 나옴
# readline_txt = file.readline()
# print(readline_txt)

# 리스트 객체로 바꿔서 나옴
readlines_txt = file.readlines()
print(readlines_txt)

file.close()