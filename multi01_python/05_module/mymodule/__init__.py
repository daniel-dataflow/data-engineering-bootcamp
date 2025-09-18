# 기본 정보 등등을 작성할 수 있다!
VERSION = "1.0.0"

# 초기화 시, 미리 생성되어야 하는 것들 (ex. database connection) 등등을 설정
def print_version():
    print(VERSION)

print_version()

# from mymoudle import * 에서 * 을 통해 접근하는 module을 설정!
__all__ = ["mytest"]

