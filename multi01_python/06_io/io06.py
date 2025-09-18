with open("test01.txt", "r", encoding="utf-8") as file :
    for line in file.readlines():
        print(line)


# with 구문은 자동으로 close 실행!