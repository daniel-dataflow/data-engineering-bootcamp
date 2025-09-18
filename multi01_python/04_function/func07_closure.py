def greetings(name):
    prefix = "안녕하세요! "

    # lexical scope: suffix 라는 함수가 선언 될때, 함수의 범위(scope)가 정해짐
    # suffix가 return 되면 greetings도 종료되지만, suffix를 실행 시 greetings의 환경(prefix, name)를 기억했다가 사용
    # 클로저 함수 :  내부에 있는 함수를 자기를 감싸고 있는 함수가 사용할 수 있어요.
    def suffix():
        def msg ():
            return prefix + name + "님!! 환영합니다!!!"

        return msg

    # 함수가 값으로 사용됨! (일급객체, 일급함수, 일급시민,....)
    return suffix

if __name__ == "__main__":
    message = greetings("한대성")()()
    print(message)