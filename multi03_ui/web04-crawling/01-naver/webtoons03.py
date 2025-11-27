# pip install chromediver_autoinstaller // 크롬 드라이버 자동으로 설치해주는
import chromedriver_autoinstaller

import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

from bs4 import BeautifulSoup
from time import sleep
import json

chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
# print(chrome_ver)

if os.path.exists(f"./{chrome_ver}"):
    print("exist")
else:
    chromedriver_autoinstaller.install(True)

target_url = "https://comic.naver.com/webtoon?tab=thu"

service = Service(executable_path=f"./{chrome_ver}/chromedriver") # 맥은 .exe 쓰면 안됨 # 폴더를 만들고 그 안에 드라이버를 설치해버린다.
# service = Service(executable_path=f"./{chrome_ver}/chromedriver.exe") # 윈도우용
driver = webdriver.Chrome(service=service) # 크롬을 실행할 수 있어요.

driver.get(target_url)

sleep(2) # 비동기 통신의 대기시간
soup = BeautifulSoup(driver.page_source, "html.parser")
# print(soup)

webtoons_list = list()
li_list  = soup.select(".component_wrap .item")
for li in li_list:
    star = li.find("span", class_="Rating__star_area--dFzsb").text
    title = li.find("span", class_="ContentTitle__title--e3qXt").text

    print(f"{star} \t {title}")
    webroon = dict()
    webroon["star"] = star[2:]
    webroon["title"] = title
    webtoons_list.append(webroon)

driver.quit()

result = dict()
result["webtoons"] = webtoons_list

result_json = json.dumps(result, ensure_ascii=False)

with open("webtoons.json", "w", encoding="utf-8") as file:
    file.write(result_json)