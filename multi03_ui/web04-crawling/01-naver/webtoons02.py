# pip install selenium
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep

target_url = "https://comic.naver.com/webtoon?tab=thu"

driver = webdriver.Chrome()

driver.get(target_url)

sleep(2) # 비동기 통신의 대기시간
soup = BeautifulSoup(driver.page_source, "html.parser")
# print(soup)

li_list  = soup.select(".component_wrap .item")
for li in li_list:
    star = li.find("span", class_="Rating__star_area--dFzsb").text
    title = li.find("span", class_="ContentTitle__title--e3qXt").text
    print(f"{star} \t {title}")





driver.quit()