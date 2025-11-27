from bs4 import BeautifulSoup
import requests

target_url = "https://comic.naver.com/webtoon?tab=thu"
resp = requests.get(target_url)
soup = BeautifulSoup(resp.text, "html.parser")
print(soup)

component_wrap = soup.find_all("div", class_="component_wrap")
print(component_wrap)