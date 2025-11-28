import streamlit as st
import pandas as pd


df11 = pd.read_csv("./data/card_data/csv/202111.csv")
df12 = pd.read_csv("./data/card_data/csv/202112.csv")

day_df11 = df11[["이용일", "금액"]]
day_df12 = df12[["이용일", "금액"]]

day_df11["이용일"] = day_df11["이용일"].map(lambda x: x-20211100)
day_df12["이용일"] = day_df12["이용일"].map(lambda x: x-20211200)

chart_df11 = day_df11.groupby("이용일").sum()
chart_df12 = day_df12.groupby("이용일").sum()

st.dataframe(chart_df11)
st.dataframe(chart_df12)

chart_df = chart_df11.merge(chart_df12, on="이용일", how="outer")
st.dataframe(chart_df)

days = pd.DataFrame(range(1, 32), columns=["일"])
chart_df = days.merge(chart_df, left_on="일", right_on="이용일", how="outer")
st.dataframe(chart_df)


chart_df.columns = ["이용일", "11월 금액", "12월 금액"]
chart_df.fillna(value=0, inplace=True)
st.dataframe(chart_df)


st.area_chart(chart_df, x="이용일", y=["11월 금액", "12월 금액"])

st.bar_chart(chart_df, x="이용일")
st.bar_chart(chart_df, x="이용일", horizontal=True)
st.bar_chart(chart_df, x="이용일", stack=True)

st.line_chart(chart_df, x="이용일")
st.line_chart(chart_df, x="이용일", color=["#ff0000", "#0000FF"])

import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)
ax[0].bar(x=chart_df["이용일"], height=chart_df["11월 금액"])
ax[1].bar(x=chart_df["이용일"], height=chart_df["12월 금액"])

st.pyplot(fig)

