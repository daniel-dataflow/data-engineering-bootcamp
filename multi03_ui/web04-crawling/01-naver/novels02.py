import urllib

from bs4 import BeautifulSoup
# import urllib.request
import requests

# pip install requests
target_url ="https://novel.naver.com/webnovel/weekday"
# resp = urllib.request.urlopen()
resp = requests.get(target_url)
# print(resp)

soup = BeautifulSoup(resp.text, "html.parser")
# print(soup)

section = soup.find("div", class_="component_section")
# print(section)

item_list = section.find_all("li", class_="item")
for item in item_list:
    # print(item)
    lank = item.p.text.strip()
    # print(lank)
    title = item.select("span.title")[0].text
    novel_url = item.a.attrs["href"]
    print(f"{lank:2}위 : {title} \t  https://novel.naver.com{novel_url}")