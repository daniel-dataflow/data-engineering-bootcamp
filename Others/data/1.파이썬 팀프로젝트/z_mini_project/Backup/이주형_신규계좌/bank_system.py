from bank_storage import BankStorage
from bank_personal import BankPersonal
from bank_account import BankAccount
from bank_interface import BankInterface
from bank_main import Main
from bank_loan import BankLoan

if __name__ == "__main__":
    storage = BankStorage("bank.json")
    per = BankPersonal(storage)
    acc = BankAccount(storage, per)
    loan = BankLoan(storage, per)
    main = Main(per, acc, loan)
    ui = BankInterface(per, acc, main, loan)
    ui.main_interface()