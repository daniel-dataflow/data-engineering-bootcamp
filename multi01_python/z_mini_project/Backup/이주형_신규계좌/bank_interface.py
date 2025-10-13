from datetime import datetime

class BankInterface:
    def __init__(self, per, acc, main, loan=None):
        self.per = per
        self.acc = acc
        self.main = main
        self.loan = loan

    # 로그인 전 인터페이스 기능
    def main_interface(self):
        while True:
            now = datetime.now()
            if self.per.current_login == 0:
                print("")
                print(f"현재 시간은 {now.month}월 {now.day}일 {now.hour}시 {now.minute}분 입니다.")
                print("-----------------------------------------")
                print("| [메뉴] 0: 회원가입 | 1: 로그인 | 9: 종료 |")
                print("-----------------------------------------")
                choice = input("번호를 입력하세요: ").strip()
                if not self.main.login_main(choice):
                    break


            else:
                print("")
                print(f"현재 시간은 {now.month}월 {now.day}일 {now.hour}시 {now.minute}분 입니다.")
                print("----------------------------------------------------")
                print("| [메뉴] 0: 회원가입 | 1: 로그아웃 | 2: 계좌 | 9: 종료 |")
                print("----------------------------------------------------")
                choice = input("번호를 입력하세요: ").strip()
                if choice == "2":
                    self.account_interface()
                else:
                    if not self.main.logout_main(choice):
                        break

    # 로그인 후 2번을 누르면 나오는 계좌 관련 인터페이스
    def account_interface(self):
        while True:
            print("")
            print("--------------------------------------------------------------------------------------------")
            print("| [계좌] 0: 조회 | 1: 개설 | 2: 삭제 | 3: 이체 | 4: 입/출금 | 5: 대출 | 8: 뒤로가기 | 9: 종료 |")
            print("--------------------------------------------------------------------------------------------")
            choice = input("번호를 입력하세요: ").strip()
            
            result = self.main.account_main(choice)
            if result == "deposit_withdraw":
                self.deposit_withdraw_interface()
            elif result == "loan":
                self.loan_interface()
            elif result == False:
                break

    # 4.입/출금을 누르면 나오는 입/출금 관련 인터페이스
    def deposit_withdraw_interface(self):
        while True:
            print("")
            print("---------------------------------------------------")
            print("| [입/출금] 0: 입금 | 1: 출금 | 8. 뒤로가기 | 9. 종료 |")
            print("---------------------------------------------------")
            choice = input("번호를 입력하세요: ").strip()
            if not self.main.deposit_withdraw_main(choice):
                break

    # 3.대출을 누르면 나오는 대출 관련 인터페이스
    def loan_interface(self):
        if not self.loan:
            print("대출 서비스가 초기화되지 않았습니다.")
            return
            
        while True:
            print("")
            print("-------------------------------------------------------------")
            print("| [대출] 0: 조회 | 1: 신청 | 2: 상환 | 8. 뒤로가기 | 9. 종료 |")
            print("-------------------------------------------------------------")
            choice = input("번호를 입력하세요: ").strip()
            if not self.main.loan_main(choice):
                break