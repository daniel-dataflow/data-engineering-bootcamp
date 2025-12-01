import streamlit as st
import pandas as pd


"""
# 월별 보기
"""

all_data = st.session_state["all_data"]

all_data["이용일"] = pd.to_datetime(all_data["이용일"].astype(str))
all_data["년도"] = all_data["이용일"].dt.year
all_data["월"] = all_data["이용일"].dt.month
all_data["일"] = all_data["이용일"].dt.day
all_data["요일"] = all_data["이용일"].dt.weekday
weekday_list = ["월", "화", "수", "목", "금", "토", "일"]
all_data["요일"] = all_data.apply(lambda x: weekday_list[x["요일"]], axis=1)

months_df = all_data[["월", "금액"]].groupby("월").mean()
st.dataframe(months_df)

st.line_chart(months_df)