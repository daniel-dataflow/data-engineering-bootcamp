class Hello:
    def __call__(self, name: str) -> str:
        print(f"Hello, {name}")


if __name__ == '__main__':
    greeting = Hello()
    # callable object :  객체를 함수처럼 사용하도록 __call__이 도와준다.- 객체 뒤에 괄호하고 다시 쓴다면
    greeting("DaeSung!")

