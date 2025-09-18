messages = ["리스트로\n", "한 번에\n", "입력하고 싶어요\n"]

file = open("test01.txt", "a", encoding="utf-8")

file.writelines(messages)

file.close()