# 숙제
# test_list 에서 숫자만 새로운 list로 만들어서 출력하자
test_list = ["3", "6", None, "9", "", "a"]
# hint! str.isdigit()
# hint!! list, filter, lambda
print(list(filter(lambda x: x if(x == None) else x.isdigit() , test_list)))
print(list(filter(lambda x: x.isdigit() if x else None  , test_list)))
