import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv("./data/card_data/csv/202112.csv")
st.write(df)

fig, ax = plt.subplots()
ax.bar(x=df["이용일"], height=df["금액"])
st.write(fig)


def lovely():
    love = "❤︎"
    for _ in range(10):
        time.sleep(0.5)
        yield love + " "

st.write_stream(lovely)

df

"""
# 이것도
## 자동으로
### 됩니다!!
"""

fig