import streamlit as st
import pandas as pd

df11 = pd.read_csv("./data/card_data/csv/202111.csv")
df12 = pd.read_csv("./data/card_data/csv/202112.csv")

chart_df11 = df11[["이용일", "금액"]]
chart_df12 = df12[["이용일", "금액"]]

left, right = st.columns(2)
left.table(chart_df11)
right.table(chart_df12)

mean_df11 = int(chart_df11["금액"].mean())
mean_df12 = int(chart_df12["금액"].mean())

left, middle, right = st.columns(3)
left.header("11월 금액")
left.bar_chart(chart_df11, x="이용일")

middle.header("12월 금액")
middle.bar_chart(chart_df12, x="이용일")

right.write(f"### 11월 금액 평균 : {mean_df11}")
right.metric(label="전월 대비", value=mean_df12, delta=mean_df12-mean_df11)



list_2021 = list()
for i in range(1, 13):
    if i < 10:
        i = f"0{i}"
    list_2021.append(pd.read_csv(f"./data/card_data/csv/2021{i}.csv"))

mean_2021 = list()
for df in list_2021:
    mean_2021.append(int(df["금액"].mean()))

mean_df_2021 = pd.DataFrame(mean_2021, columns=["평균금액"])
mean_df_2021.insert(0, "월", range(1, 13))

mean_2021 = int(mean_df_2021["평균금액"].mean())

left, right = st.columns([2, 1])
left.line_chart(mean_df_2021, x="월")
right.write(f"2021년도의 평균금액 : **{mean_2021}**")



left, middle, right = st.columns([3, 1, 1])
vertical_alignment = left.selectbox("vertical alignment", ["top", "center", "bottom"])
gap = middle.selectbox("gap", ["small", "medium", "large"])
border = right.checkbox("border")

left, right = st.columns([2, 1], vertical_alignment=vertical_alignment, gap=gap, border=border)

with left:
    st.line_chart(mean_df_2021, x="월")
with right:
    st.write(f"2021년도의 평균 금액: **{mean_2021}**")



container01 = st.container(height=100)
container_list = st.columns(2)
container02, container03 = container_list

with container01:
    st.write(f"""
    container01 의 타입 : {type(container01)}\n
    columns(2) 의 타입 :  {type(container_list)}\n
    container02 의 타입 : {type(container02)}\n
    """)

container02.write("container 02 입니다.")
container03.write("container 03 입니다.")
container01.write("container 01 입니다." * 100)




from time import sleep

st.button("사랑합니다.")

with st.empty():
    for i in range(10):
        if i % 2 == 1:
            st.write("❤️")
        else:
            st.write("💙")
        sleep(0.5)





st.bar_chart(chart_df11, x="이용일")

with st.expander("11월 테이블"):
    st.write(chart_df11)






with st.popover(""):
    food = st.text_input("먹고 싶은 점심 메뉴는 ??")
    select = st.selectbox(label="위치", options=["top_left", "top_right", "bottom_left", "bottom_right"])

top_left, top_right = st.columns(2, border=True)
bottom_left, bottom_right = st.columns(2, border=True)

match select:
    case "top_left":
        top_left.write(food)
    case "top_right":
        top_right.write(food)
    case "bottom_left":
        bottom_left.write(food)
    case "bottom_right":
        bottom_right.write(food)

