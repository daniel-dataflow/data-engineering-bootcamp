import urllib
from bs4 import BeautifulSoup
# import urllib.request
import requests
import json

# pip install requests
target_url ="https://novel.naver.com/webnovel/weekday"
# resp = urllib.request.urlopen()
resp = requests.get(target_url)
# print(resp)

soup = BeautifulSoup(resp.text, "html.parser")
# print(soup)

section = soup.find("div", class_="component_section")
# print(section)

novel_list = list()
item_list = section.find_all("li", class_="item")
for item in item_list:
    # print(item)
    rank = item.p.text.strip()
    # print(lank)
    title = item.select("span.title")[0].text
    novel_url = item.a.attrs["href"]
    # print(f"{lank:2}위 : {title} \t  https://novel.naver.com{novel_url}")
    tmp = dict()
    tmp["rank"] = rank
    tmp["title"] = title
    tmp["url"] = novel_url

    novel_list.append(tmp)

result = dict()
result["novels"] = novel_list

result_json = json.dumps(result, ensure_ascii=False)

with open("novels.json", "w", encoding="utf-8") as f:
    f.write(result_json)