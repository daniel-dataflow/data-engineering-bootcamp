import requests
from xml.etree import ElementTree

BASE_URL = "https://apis.data.go.kr/1262000/TravelWarningServiceV3"
SERVICE_KEY = "02044344ad4f839f458ce0ad728e91cb82428df574fab7d27ab246671d8872b4"
CALL_URL = "/getTravelWarningListV3"

url = f"{BASE_URL}{CALL_URL}?serviceKey={SERVICE_KEY}&pageNo=1&numOfRows=10&returnType=xml"
# print(url)

resp = requests.get(url)
# print(resp.text) # 문자열

tree = ElementTree.fromstring(resp.text)
# print(tree)
# print(tree[0][1])

for item in tree[0][1]:
    if item.find("attention").text == "여행유의":
        print(item.find("country_name").text)
