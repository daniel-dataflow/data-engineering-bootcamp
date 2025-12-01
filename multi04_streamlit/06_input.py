import streamlit as st
from altair.theme import enable

st.button("그냥 버튼")

st.button("안녕하세요.", type="primary")
st.button("어서오세요.", type="tertiary")

if st.button("click me!"):
    st.write("clicked")


if st.button("click me?"):
    st.write("!!!")
else:
    st.write("???")

def test():
    st.write("it's test function")

st.button("테스트", on_click=test)

def layout_test():
    layout.write("layout test!")

layout = st.columns(1)[0]
layout.button("레이아웃 테스트!", on_click=layout_test)

import pandas as pd

@st.cache_data
def csv_download():
    return pd.read_csv("./data/card_data/csv/202112.csv").to_csv().encode("utf-8")

csv = csv_download()
st.download_button(
    label="download csv",
    data=csv,
    file_name="202112.csv",
    mime="text/csv"
)


st.link_button("google", "https://google.com")

check = st.checkbox("성인입니다.")

if check:
    st.write("성인이시군요")

gender = st.checkbox("남성입니다.")
if gender:
    st.write("남성")
else:
    st.write("여성")


radio = st.radio("성별을 알려주세요.", ["남성", "여성"])
if radio == "남성":
    st.write("남성이시군요.")
else:
    st.write("여성이시군요.")


food = ["돼지고기", "소고기", "물냉면", "볶음밥", "계란찜"]
options = st.multiselect(
    "좋아하는 음식을 선택해 주세요.",
    food,
    accept_new_options=True
)

options = st.selectbox(
    "좋아하는 음식을 골라주세요.",
    food
)

on = st.toggle("음악 실행??")
if on:
    st.write("음악 실행중")
else:
    st.write("음악 재생 준비")


sentiment_mapping = ["매우 불만족", "불만족", "보통", "만족", "매우 만족"]
selected = st.feedback("faces")
if selected is not None:
    st.write(f"{sentiment_mapping[selected]} 을 선택하셨습니다.")

number = st.number_input("숫자를 입력해 주세요.", value=0)
st.write(f"현재 입력된 숫자는 {number} 입니다...")

number = st.number_input("숫자를 입력해 주세요.")
st.write(f"현재 입력된 숫자는 {number} 입니다...")

number = st.number_input("숫자를 입력해 주세요.", step=1)
st.write(f"현재 입력된 숫자는 {number} 입니다...")

import datetime

birthday = st.date_input("너의 생일은...", value=datetime.date.today())
st.write(f"당신의 생일은 {birthday} 이군요!!")

start_day = datetime.date(2025, 9, 15)
end_day = datetime.date(2026, 3, 13)
study = st.date_input("우리의 공부기간은...", (start_day, end_day))

resume = st.text_area("자소서",  "저는 인자하신 아버지와 자상하신 어머니 슬하의 막내로 태어나 ...\n"
                        "이런 느낌으로 자소서를 작성하시면 \n"
                        "탈락입니다.")
st.write(f"총 {len(resume)} 단어")


food  = st.text_input("좋아하는 음식은?")
st.write(f"내가 좋아하는 음식은 {food} !!!")

st.image("resources/img01.png")
st.audio("resources/a.mp3")

audio_value = st.audio_input("오디오 입력 기능 (브라우저에서 허용 필수!")
if audio_value:
    st.audio(audio_value)

st.video("https://youtu.be/Qe8fa4b5xNU?si=J-gJNfxFhXZ3jyqX")

enable == st.checkbox("카메라 입력")
picture = st.camera_input("take a picture", disabled=not enable)
if picture:
    st.image(picture)