import json


"""
json 저장 예시:
{
  "users": [
    {
      "id": "....",
      "password": ".....",
      "accounts": ["....", "...."]
    }
  ]
}
"""
DATA_FILE = "bank.json"

# 데이터 불러오기
def load_db():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": []}
    except json.JSONDecodeError:
        return {"users": []}

# 데이터 저장하기
def save_db(db):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# id가 있으면 해당 dict를 반환, 아니면 None 반환
def find_user(db, user_id):
    for u in db["users"]:
        if u["id"] == user_id:
            return u
    return None

# 회원가입 기능
def join_the_membership():
    db = load_db()

    name = input("가입할 이름을 입력하세요: ")
    user_id = input("가입할 ID를 입력하세요: ")

    # 가입할 id가 이미 있으면)
    if find_user(db, user_id):
        print("이미 존재하는 ID입니다.")
        return False

    pw = input("가입할 PW를 입력하세요: ")
    ac = input("가입할 계좌번호를 입력하세요: ")

    db["users"].append({
        "name": name,
        "id": user_id,
        "password": pw,  # 학습용: 평문 저장
        "accounts": [ac]  # 리스트 형태
    })

    save_db(db)
    print("가입완료!")
    return True

# 로그인 기능
def login():
    db = load_db()

    user_id = input("아이디를 입력하세요: ")
    pw = input("패스워드를 입력하세요: ")

    user = find_user(db, user_id)
    # ID 있는지/맞았는지 확인!
    if not user:
        print("가입된 ID가 없거나 ID가 틀렸습니다. 다시 시도해주세요.")
        return False

    # PW 있는지/맞았는지 확인!
    if user["password"] != pw:
        print("비밀번호가 틀렸습니다. 다시 시도해주세요.")
        return False

    print(f"로그인 성공! {user['name']} 님 환영합니다!")

    # current_login에 1 반환
    return 1

# 로그아웃 기능
def logout():
    print("로그아웃 되었습니다!")
    return 0


# 로그인 전 인터페이스 기능
def interface():
    current_login = 0
    while True:
        if current_login == 0:
            print("[메뉴] 0: 회원가입 | 1: 로그인 | 9: 종료")
            choice = input("번호를 입력하세요: ")

            if choice == "0":
                join_the_membership()
            elif choice == "1":
                current_login = login()
            elif choice == "9":
                print("종료합니다.")
                break
            else:
                print("올바른 번호를 입력하세요.")

        else:
            print("[메뉴] 0: 회원가입 | 1: 로그아웃 | 9: 종료")
            choice = input("번호를 입력하세요: ")

            if choice == "0":
                join_the_membership()
            elif choice == "1":
                current_login = logout()
            elif choice == "9":
                print("종료합니다.")
                break
            else:
                print("올바른 번호를 입력하세요.")

if __name__ == "__main__":
    interface()