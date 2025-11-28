from unicodedata import category

import streamlit as st
import pandas as pd
from altair.theme import options

df = pd.read_csv("./data/card_data/csv/202112.csv")
st.dataframe(df)


st.dataframe(df.style.highlight_null("gray"))

st.dataframe(df.style.highlight_max(subset="금액"))

# hide 안먹힘
st.dataframe(df.style.hide(subset=["결제일", "포인트리금액"], axis=1))

st.dataframe(df, column_config={"금액": st.column_config.NumberColumn("금액", format="%d won")})

df["이용일"] = pd.to_datetime(df["이용일"].astype("str"))
st.dataframe(df, column_config={"이용일": st.column_config.DateColumn("이용일", format="DD MMM YYYY")})

if "df" not in st.session_state:
    st.session_state.df = df

event = st.dataframe(
    st.session_state.df,
    key="data",
    on_select="rerun",
    selection_mode=["multi-row", "multi-column"]
)

st.dataframe(df) # 동적
st.table(df) # 정적


# data_edito로 날짜를 수정할 수 있다.
from datetime import datetime

st.data_editor(
    df,
    column_config={
        "이용일": st.column_config.DatetimeColumn(
            "이용일",
            min_value=datetime(2021, 1, 1),
            max_value=datetime(2021, 12, 31),
            format="DD MMM YYYY"
        )
    },
    disabled=False
)

# 대분류에 셀렉트 박스로 리스트 항목 만들어준다.
category_list = list(df["대분류"])
st.data_editor(
    df,
    column_config={
        "대분류":st.column_config.SelectboxColumn(
            "대분류",
            options=category_list,
            required=True
        )
    },
    hide_index=True
)

select_list = df["소분류"].dropna().to_list()
st.data_editor(
    df,
    column_config={
        "대분류": st.column_config.SelectboxColumn(
        "대분류",
        options=category_list,
        required=True
        ),
        "소분류": st.column_config.SelectboxColumn(
        "소분류",
        options=select_list
        )
    }
)

amount_df = pd.DataFrame({"amount": [df["금액"].tolist()]})
st.dataframe(amount_df)

amount_min = min(amount_df["amount"].tolist()[0])
amount_max = max(amount_df["amount"].tolist()[0])

st.data_editor(
    amount_df,
    column_config={
        "amount": st.column_config.LineChartColumn(
            "amount",
            y_min=amount_min,
            y_max=amount_max
        )
    }
)

st.data_editor(
    amount_df,
    column_config={
        "amount": st.column_config.BarChartColumn(
            "amount",
            y_min=amount_min,
            y_max=amount_max
        )
    }
)

# 이상치는 제거
amount_list = amount_df["amount"][0]
amount_list.remove(max(amount_list))

# 수치를 작게 줄여본다.
amount_list = list(map(lambda x: x /1000, amount_list))
amount_df.loc[0, "amount"] = amount_list

st.dataframe(amount_df)

amount_min = min(amount_df["amount"].tolist()[0])
amount_max = max(amount_df["amount"].tolist()[0])

st.data_editor(
    amount_df,
    column_config={
        "amount": st.column_config.LineChartColumn(
            "amount",
            y_min=amount_min,
            y_max=amount_max
        )
    }
)

st.data_editor(
    amount_df,
    column_config={
        "amount": st.column_config.BarChartColumn(
            "amount",
            y_min=amount_min,
            y_max=amount_max
        )
    }
)

df11 = pd.read_csv("./data/card_data/csv/202111.csv")
df12 = pd.read_csv("./data/card_data/csv/202112.csv")

mean_df11 = int(df11["금액"].mean())
mean_df12 = int(df12["금액"].mean())

st.metric(label="전원대비", value=mean_df12, delta=mean_df12 - mean_df11)
st.metric(label="전원대비", value=mean_df12, delta=mean_df12 - mean_df11, border=True)
st.metric(label="전원대비", value=mean_df12, delta=mean_df12 - mean_df11, delta_color="inverse")