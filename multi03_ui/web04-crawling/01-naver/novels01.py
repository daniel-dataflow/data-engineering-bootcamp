from bs4 import BeautifulSoup
import urllib.request

resp = urllib.request.urlopen("https://novel.naver.com/webnovel/weekday")
# print(resp)
soup = BeautifulSoup(resp, "html.parser")
# print(soup)

section = soup.find("div", class_="component_section")
# print(section)

item_list = section.find_all("li", class_="item")
for item in item_list:
    # print(item)
    lank = item.p.text.strip()
    # print(lank)
    title = item.select("span.title")[0].text
    print(f"{lank:2}위 : {title}")