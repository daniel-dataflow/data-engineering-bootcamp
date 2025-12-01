import streamlit as st
import pandas as pd


"""
# 년도별 보기
"""

all_data = st.session_state["all_data"]

all_data["이용일"] = pd.to_datetime(all_data["이용일"].astype(str))
all_data["년도"] = all_data["이용일"].dt.year
all_data["월"] = all_data["이용일"].dt.month
all_data["일"] = all_data["이용일"].dt.day
all_data["요일"] = all_data["이용일"].dt.weekday
weekday_list = ["월", "화", "수", "목", "금", "토", "일"]
all_data["요일"] = all_data.apply(lambda x: weekday_list[x["요일"]], axis=1)

years_df = all_data[["년도", "월", "금액"]].groupby(["년도", "월"]).mean()
years_df.reset_index(level=['년도', "월"], inplace = True)
st.dataframe(years_df)

years_df = years_df.astype(dtype="int64")
st.line_chart(years_df, x="월", y="금액", color="년도")
