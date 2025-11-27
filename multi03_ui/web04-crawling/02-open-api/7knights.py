import requests
from os import path
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from chromedriver_autoinstaller import get_chrome_version, install


uids = """
uid_1234
"""

coupons ="""
coupon_1234
"""

"""
pip install chromedriver_autoinstaller
pip install selenium
pip install requests
"""
class SevenKnightsCoupon:

    def __init__(self, headless=True):
        self.coupon_info = ""

        try:
            chrome_ver = get_chrome_version().split(".")[0]
            if path.exists(f'./{chrome_ver}'):
                pass
            else:
                install(True)

            # service = Service(f'./{chrome_ver}/chromedriver.exe') # 윈도우용
            service = Service(f'./{chrome_ver}/chromedriver')  # 맥용

            if headless:
                options = webdriver.ChromeOptions()
                options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                self.browser = webdriver.Chrome(service=service, options=options)
            else:
                self.browser = webdriver.Chrome(service=service)

        except Exception as e:
            print(e)
            print('please driver check')


    def coupon_data(self, uid, coupon):
        # 쿠폰 정보 출력
        coupon_data_url = f"https://coupon.netmarble.com/api/coupon/reward?gameCode=tskgb&couponCode={coupon}&langCd=KO_KR&pid={uid}"
        resp = requests.get(coupon_data_url)
        resp_dict = resp.json()

        # 응답 내용 확인 (json 구성이 제각각임)
        # print(resp_dict)

        try:
            if "groupName" in resp_dict.keys():
                self.coupon_info = resp_dict["groupName"]
            else:
                for data in resp_dict["resultData"]:
                    if "productName" in data.keys():
                        self.coupon_info = data["productName"] + " "
                    elif "products" in data.keys():
                        self.coupon_info = data["products"]["productName"] + " "
        except:
            self.coupon_info = resp_dict["errorMessage"]

        return self


    def coupon_input(self, input_uids, input_coupons):
        coupon_url = "https://coupon.netmarble.com/tskgb"
        self.browser.get(coupon_url)

        uid_list = list(filter(lambda x: x, input_uids.split("\n")))
        coupon_list = list(filter(lambda x: x, input_coupons.split("\n")))

        for uid in uid_list:
            print(uid)

            for coupon in coupon_list:
                self.coupon_data(uid, coupon)
                print(f"{coupon} : {self.coupon_info}", end="")

                input_text_list = self.browser.find_elements(By.CSS_SELECTOR, ".InputText_input__l1Hgv")
                # uid 입력
                input_text_list[0].send_keys(uid)
                # coupon 입력
                input_text_list[1].send_keys(coupon)
                # 사용하기 클릭
                self.browser.find_elements(By.CSS_SELECTOR, ".Button_button__k4Gkp")[0].click()
                sleep(0.1)
                # 확인 클릭
                self.browser.find_elements(By.CSS_SELECTOR, ".Button_button__k4Gkp")[2].click()
                sleep(0.1)
                # 새로고침
                self.browser.refresh()
                print()

            print()
            sleep(0.1)

        return self


    def __del__(self):
        self.browser.close()
        print("closed")


if __name__ == "__main__":
     SevenKnightsCoupon().coupon_input(uids, coupons)


