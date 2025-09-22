class HelloFields:

    def __init__(self, name = "홍길동", age = 100):
        # 속성 (fields)
        self.name = name
        self.age = age

    # 명명규칙
    # camel case (클래스) : 소문자, 새로운 단어가 나타나면 대문자로 시작시킨다.(pascal case : 첫글자도 대문자, 새로운 문자가 나타나면 다시 대문자)
    # snake case (함수) : 소문자, 새로운 단어가 나타나면 _로 시작해서 붙여준다.
    # kebab case (URL) : 소문자, 새로운 단어가 나타나면 -로 시작해서 붙여준다.
    # pythond은 변수, 함수(매서드), module는 camel case
    # class는 pascal case
    # 상수 :  모든 문자를 대문자, snake case (Max_LENGTH)
    def member_info(self):
        return f"이름 : {self.name} \t 나이 : {self.age}"


if __name__ == "__main__":
    class03 = HelloFields()
    print(class03.member_info())

    dongheon =  HelloFields("dongheon", 100)
    print(dongheon.member_info())
    # 변수 추가
    # instance variable 추가된거다!
    dongheon.addr = "수원시"
    print(f"{dongheon.member_info()} \t 지역 : {dongheon.addr}")

    # 다른 객체에서 사용 불가능
    # print(f"{class03.member_info()} \t 지역 : {class03.addr}")