import requests

BASE_URL = "https://apis.data.go.kr/1262000/TravelWarningServiceV3"
SERVICE_KEY = "02044344ad4f839f458ce0ad728e91cb82428df574fab7d27ab246671d8872b4"
CALL_URL = "/getTravelWarningListV3"

url = f"{BASE_URL}{CALL_URL}?serviceKey={SERVICE_KEY}&pageNo=1&numOfRows=10"
# print(url)

resp = requests.get(url)
# print(resp.text)
# print(resp.json())

# 파이썬이 json을 딕셔너리로 받을 수 있도록 되어 있다.
for item in resp.json()["response"]["body"]["items"]["item"]:
    # print(item)
    if item["attention"] == "여행유의":
        print(item["country_name"])