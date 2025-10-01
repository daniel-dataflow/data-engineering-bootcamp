import json


"""
json 저장 예시:
{
  "users": [
    {
      "name" : "...."
      "id": "....",
      "password": ".....",
      "accounts": [{"....": 0,000,  "...." : 0,000}]
    }
  ]
}
"""

# 로드/세이브 기능 구현 클래스
class BankStorage:
    def __init__(self, path="bank.json"):
        self.path=path

    # 데이터 불러오기
    def load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"users": []}

    # 데이터 저장하기
    def save(self, db):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)

# 메인 시스템 구현 클래스
class BankSystem:
    def __init__(self, storage: BankStorage):
        self.storage = storage
        self.db = self.storage.load()
        self.current_login = 0


    # id가 있으면 해당 dict를 반환, 아니면 None 반환
    def find_user(self, user_id):
        for u in self.db["users"]:
            if u["id"] == user_id:
                return u
        return None

    # 회원가입 기능
    def join(self):
        name = input("가입할 이름을 입력하세요: ")
        user_id = input("가입할 ID를 입력하세요: ")

        # 가입할 id가 이미 있으면)
        if self.find_user(user_id):
            print("이미 존재하는 ID입니다.")
            return False

        pw = input("가입할 PW를 입력하세요: ")
        ac = input("가입할 계좌번호를 입력하세요: ")

        self.db["users"].append({
            "name": name,
            "id": user_id,
            "password": pw,  # 학습용: 평문 저장
            "accounts": [{ac : 0}]  # 리스트 형태
        })

        self.storage.save(self.db)
        print("가입완료!")
        return True

    # 로그인 기능
    def login(self):
        user_id = input("아이디를 입력하세요: ")
        pw = input("패스워드를 입력하세요: ")

        user = self.find_user(user_id)
        # ID 있는지/맞았는지 확인!
        if not user:
            print("가입된 ID가 없거나 ID가 틀렸습니다. 다시 시도해주세요.")
            return False

        # PW 있는지/맞았는지 확인!
        if user["password"] != pw:
            print("비밀번호가 틀렸습니다. 다시 시도해주세요.")
            return False

        # current_login에 1 반환
        self.current_login = 1
        print(f"로그인 성공! {user['name']} 님 환영합니다!")
        return True

    # 로그아웃 기능
    def logout(self):
        if self.current_login:
            print("로그아웃 되었습니다!")
        self.current_login = 0
        return True

class BankInterface:
    def __init__(self, system : BankSystem):
        self.system = system

    # 로그인 전 인터페이스 기능
    def interface(self):
        while True:
            if self.system.current_login == 0:
                print("[메뉴] 0: 회원가입 | 1: 로그인 | 9: 종료")
                choice = input("번호를 입력하세요: ")

                if choice == "0":
                    self.system.join()
                elif choice == "1":
                    self.system.login()
                elif choice == "9":
                    print("종료합니다.")
                    break
                else:
                    print("올바른 번호를 입력하세요.")

            else:
                print("[메뉴] 0: 회원가입 | 1: 로그아웃 | 9: 종료")
                choice = input("번호를 입력하세요: ")

                if choice == "0":
                    self.system.join()
                elif choice == "1":
                    self.system.logout()
                elif choice == "9":
                    print("종료합니다.")
                    break
                else:
                    print("올바른 번호를 입력하세요.")

if __name__ == "__main__":
    storage = BankStorage("bank.json")
    app = BankSystem(storage)
    ui = BankInterface(app)
    ui.interface()