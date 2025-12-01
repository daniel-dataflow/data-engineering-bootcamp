import streamlit as st
from streamlit import columns


@st.dialog("주소록 입력")
def address_input():
    columns = [("name", "이름"), ("address", "주소"), ("phone", "전화번호")]

    for key, val in columns:
        left, right = st.columns([1, 4])
        left.write(val)
        right.text_input(label=key, key=key)

    if st.button("주소 입력"):
        for key in st.session_state:
            address_data()[key].append(st.session_state[key])

        st.rerun()

@st.cache_resource # st.cache 데이터에서 저장할 수 있는 것 이외 데이터를 저장
def address_data():
    return {"name" :[], "address" :[], "phone" :[]}

st.dataframe(address_data())

if st.button("입력"):
    address_input()


import time

def lovely():
    love = "❤️"
    for _ in range(10):
        time.sleep(1)
        yield love + " "

if st.button("stop!"):
    st.stop()
else:
    st.write_stream(lovely)

import pandas as pd

@st.cache_data
def get_data(year, month):
    if month <10:
        month = f"0{month}"

    df = pd.read_csv(f"data/card_data/csv/{year}{month}.csv")
    return df

# st.fragment : 내부 안에 있는 내용만 바꿔주세요
@st.fragment
def tabel_month_data():
    left_left, left_right = st.columns(2)
    year_select = left_left.selectbox(label="year_select", options=year)
    month_select = left_right.selectbox(label="month_select", options=month)

    st.session_state.year = year_select
    st.session_state.month = month_select

    df = get_data(year_select, month_select)
    st.write(df)

def line_month_data():
    if st.button("라인 그래프"):
        year = st.session_state["year"]
        month = st.session_state["month"]

        st.write(f"{year} 년 {month} 월")
        df = get_data(year, month)[["이용일", "금액"]]
        st.line_chart(df, x="이용일")

left01, right01 = st.columns(2)
left01.header("월별 데이터 확인")

left02, right02 = st.columns(2)

year = [i for i in range(2019, 2022)]
month = [i for i in range(1, 13)]

with left02.container():
    tabel_month_data()

with right02.container():
    line_month_data()



