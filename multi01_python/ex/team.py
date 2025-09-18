from random import shuffle


member = """
강지연
곽동원
김민지
김소영
김예지
박주언
안호용
이주형
이창주
이한영
정수아
한대성
허무지
"""

member_list = list(filter(lambda x: x , member.split("\n")))
# print(member_list)
shuffle(member_list)
# print(member_list)

print(f"team 1 : {member_list[:4]}")
print(f"team 2 : {member_list[4:8]}")
print(f"team 3 : {member_list[8:]}")

