from selenium import webdriver
from time import sleep

target_url = "https://shoppinglive.naver.com/search/lives?query=%ED%82%A4%EB%B3%B4%EB%93%9C"

diver = webdriver.Chrome()
diver.get(target_url)
sleep(2)

for i in range(10):
    # js window 객체가 가진 scrollBy(x좌표, y좌표) 사용
    diver.execute_script(f"scrollBy(0, 1000);")
    sleep(1)

diver.quit()