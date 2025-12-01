import streamlit as st
import matplotlib.pyplot as plt

"""
# 분류별 보기
"""

all_data = st.session_state["all_data"]
chart_data = all_data[["이용일", "금액", "대분류", "중분류", "소분류"]]

category_container, chart_container = st.columns([1, 4], vertical_alignment="center")
bar_container = st.container()

def return_checked_box(checked_list):
    checked_cate = list()
    for cate, checked in checked_list:
        if checked:
            checked_cate.append(cate)

    return checked_cate

def checked_pie_chart(checked_cate):
    pie_data = chart_data[chart_data["대분류"].isin(checked_cate)]
    pie_data = pie_data["대분류"].value_counts()

    plt.rcParams["font.family"] = "AppleGothic"

    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
    ax.set_title("카드 이용 횟수")
    chart_container.pyplot(fig)

def checked_bar_chart(checked_cate):
    bar_data = chart_data[chart_data["대분류"].isin(checked_cate)]

    mean_data = bar_data[["대분류", "금액"]].groupby(by="대분류").mean()
    mean_data["금액"] = mean_data["금액"].map(int)
    bar_container.bar_chart(mean_data, x_label="대분류", y_label="금액 평균")

    max = int(mean_data["금액"].max())
    bar_container.data_editor(
        mean_data,
        column_config={
            "금액": st.column_config.ProgressColumn("금액 평균", min_value=0, max_value=max, format="%d원")
        }
    )

with category_container:
    category = chart_data["대분류"].unique()

    checked_list = list()
    for cate in category:
        chceched_box = st.checkbox(label=cate, value=True)
        checked_list.append((cate, chceched_box))

    checked_cate = return_checked_box(checked_list)

    checked_pie_chart(checked_cate)
    checked_bar_chart(checked_cate)

st.divider()

middle_cate_container, store_container = st.tabs(["중분류", "소분류"])
with middle_cate_container:
    category = chart_data["대분류"].unique()

    chart_data["중분류"] = chart_data["중분류"].str.split("").str.get(0)

    cate_select_container, middle_cate_container = st.columns([1, 4])
    with cate_select_container:
        cate_select = cate_select_container.selectbox("대분류", category)

    with middle_cate_container:
        middle_df = chart_data[chart_data["대분류"] == cate_select]
        middle_df["중분류"].fillna(cate_select, inplace=True)

        middle_mean = middle_df[["중분류", "금액"]].groupby("중분류").mean().map(int)
        st.bar_chart(middle_mean, horizontal=True)

with store_container:
    st.write(chart_data[chart_data["대분류"] == cate_select][["중분류", "소분류"]].groupby("중분류").value_counts())







