import streamlit as st
import pandas as pd
import time


def get_csv01():
    mylist = list()
    for i in range(19, 22):
        for j in range(1, 13):
            if j < 10:
                j = f"0{j}"
            mylist.append(pd.read_csv(f"data/card_data/csv/20{i}{j}.csv"))

    return mylist

@st.cache_data
def get_csv02():
    mylist = list()
    for i in range(19, 22):
        for j in range(1, 13):
            if j < 10:
                j = f"0{j}"
            mylist.append(pd.read_csv(f"data/card_data/csv/20{i}{j}.csv"))

    return mylist

start01 = time.time()
list01 = get_csv01()
end01 = time.time()
time01 = end01 - start01

start02 = time.time()
list02 = get_csv01()
end02 = time.time()
time02 = end02 - start02

start03 = time.time()
list03 = get_csv02()
end03 = time.time()
time03 = end03 - start03

start04 = time.time()
list04 = get_csv02()
end04 = time.time()
time04 = end04 - start04

st.write(f"""
    @st.cache_data 없이 첫번째 호출 : {time01}  \n
    @st.cache_data 없이 두번째 호출 : {time02}  \n
    \n
    @st.cache_data 추가 후 첫번째 호출 : {time03}  \n
    @st.cache_data 추가 후 두번째 호출 : {time04}  \n
""")

st.cache_data.clear()




st.write(st.session_state.keys())
st.session_state.name = "Daniel"
st.session_state["gender"] = "male"

st.write(st.session_state.keys())

st.write(f"안녕하세요 {st.session_state.name} 님!!!")

# st.write(f"저는 {st.session_state.age}세 입니다.")


st.text_input(label="txt1", key="txt1")
st.write(st.session_state.txt1)



def txt():
    container.write(st.session_state.txt2)

container = st.container()
container.text_area(label="txt2", key="txt2", on_change=txt)



st.selectbox("좋아하는 음식은??", ["빵", "마제소바", "삼겹살", "꼬리곰탕"], key="food", index=None)
st.write(st.session_state.food)



st.button("click me!", key="btn")
st.write(f"버튼 클릭 여부: {st.session_state.btn}")


st.write(st.session_state.keys())
del st.session_state.btn

st.write(st.session_state.keys())

for key in st.session_state.keys():
    del st.session_state[key]

st.write(st.session_state.keys())







