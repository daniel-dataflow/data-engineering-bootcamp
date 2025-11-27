from bs4 import BeautifulSoup
import requests


target_url = "https://www.data.go.kr/tcs/dss/selectDataSetList.do?dType=FILE&keyword=&detailKeyword=&publicDataPk=&recmSe=N&detailText=&relatedKeyword=&commaNotInData=&commaAndData=&commaOrData=&must_not=&tabId=&dataSetCoreTf=&coreDataNm=&sort=updtDt&relRadio=&orgFullName=&orgFilter=&org=&orgSearch=&currentPage=1&perPage=10&brm=%EA%B5%90%EC%9C%A1&instt=&svcType=&kwrdArray=&extsn=&coreDataNmArray=&operator=AND&pblonsipScopeCode=PBDE07"

resp = requests.get(target_url)
soup = BeautifulSoup(resp.text, "html.parser")
paging = soup.find("nav",{"class":"pagination"})
# print(paging)

nums = list(filter(None, map(lambda x: x.text if x.text.isdigit() else None, paging)))
# print(nums)

for i in nums:
    sub_url = "https://www.data.go.kr/tcs/dss/selectDataSetList.do?dType=FILE&keyword=&detailKeyword=&publicDataPk=&recmSe=N&detailText=&relatedKeyword=&commaNotInData=&commaAndData=&commaOrData=&must_not=&tabId=&dataSetCoreTf=&coreDataNm=&sort=updtDt&relRadio=&orgFullName=&orgFilter=&org=&orgSearch=&currentPage=" + i
    sub_soup = BeautifulSoup(requests.get(sub_url).text, "html.parser")
    titles = soup.select("span[class=recent-title]")
    for title in titles:
        print(title.text.strip().replace("\n", ""))