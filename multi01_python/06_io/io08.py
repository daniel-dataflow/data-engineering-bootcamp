import pickle


with open("test02.txt", "rb") as file:
    score = pickle.load(file)
    print(score)

# with 구문 안에서 만들어진 변수도 외부에서 사용 가능! 전역변수는 아님(함수생성으로 확인해봄)
print(score)