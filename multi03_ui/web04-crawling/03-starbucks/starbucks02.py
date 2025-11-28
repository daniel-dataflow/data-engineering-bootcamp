import requests
import json

def getSiDo():
    url ="https://www.starbucks.co.kr/store/getSidoList.do" # 이벤트가 일어나는 시점을 찾아서 작성해준다.
    resp = requests.post(url)
    # print(resp.text)

    sido_json = resp.json()["list"]
    sido_code = list(map(lambda x: x["sido_cd"], sido_json))
    sido_name = list(map(lambda x: x["sido_nm"], sido_json))

    sido_dict = dict(zip(sido_code, sido_name))

    return sido_dict

def getGugun(sido_code):
    url = "https://www.starbucks.co.kr/store/getGugunList.do"
    resp = requests.post(url, data={"sido_cd":sido_code})
    # print(resp.text)

    gugun_json = resp.json()["list"]
    gugun_dict = dict(zip(list(map(lambda x:x["gugun_cd"], gugun_json)),
                          list(map(lambda x:x["gugun_nm"], gugun_json))))

    return gugun_dict


def getStore(sido_code="", gugun_code=""):
    url = "https://www.starbucks.co.kr/store/getStore.do"
    resp = requests.post(url, data={
        "ins_lat": "37.56682",
        "ins_lon": "126.97865",
        "p_sido_cd": sido_code,
        "p_gugun_cd": gugun_code,
        "in_biz_cd": "",
        "set_date": ""
    })

    # print(resp.text)

    store_json = resp.json()["list"]

    store_list = list()
    count = 0
    for store in store_json:
        store_dict = dict()
        store_dict["s_name"] = store["s_name"]
        store_dict["doro_address"] = store["doro_address"]
        store_dict["lat"] = store["lat"]
        store_dict["lot"] = store["lot"]

        store_list.append(store_dict)
        count += 1

    # store_result = dict()
    # store_result["list"] = store_list
    # store_result["count"] = count
    #
    # result = json.dumps(store_result, ensure_ascii=False)

    return store_list

if  __name__ == "__main__":
    # print(getSiDo())
    # sido = input("도시 코드를 입력해 주세요")
    #
    # if sido == "17":
    #     print(getStore(17, 1701))
    # else:
    #     print(getGugun(sido))
    #     gugun = input("구군 코드를 입력해 주세요 : ")
    #     print(getStore(sido, gugun))

##########################################################
    list_all = list()

    for sido in getSiDo():
        if sido == "17":
            result = getStore(sido_code=sido)
            print(result)
            list_all.extend(result)
        else:
            for gugun in getGugun(sido):
                result = getStore(sido_code=sido, gugun_code=gugun)
                print(result)
                list_all.extend(result)

    print(len(list_all))
    store_result = dict()
    store_result["list"] = list_all
    result = json.dumps(store_result, ensure_ascii=False)

    with open("starbucks.json", "w", encoding="utf-8") as f:
        f.write(result)

    #  {"lsit": [{"s_name":"", "doraddress":"", "lat":"", "lot":""},
    #           [{"s_name":"", "doraddress":"", "lat":"", "lot":""},
    #           [{"s_name":"", "doraddress":"", "lat":"", "lot":""},
    #           [{"s_name":"", "doraddress":"", "lat":"", "lot":""},
    #           ...]}
    # 형태로 starbucks.json이 만들어지도록 비어있는 부분을 코드로 채우자!
