import streamlit as st
from streamlit import empty, container

message = st.selectbox(label="message", options=["success", "info", "warning", "exception"],
                       placeholder="message를 선택해주세요", index=None)

match message:
    case "success":
        st.success("성공!!")
    case "info":
        st.info("정보!!")
    case "warning":
        st.warning("경고!!")
    case "exception":
        e = Exception("예외!!")
        st.exception(e)




from time import sleep

# empty : 아무것도 없는 container
empty = st.empty()
for i in range(0, 101, 25):
    empty.progress(i, f"{i}% 진행중...")
    sleep(1.5)
    if i == 100:
        empty.success("설치 완료!!!")



container = st.progress(0, "설치 준비중...")
sleep(1)
for i in range(101):
    container.progress(i, f"{i}% 진행중")
    sleep(0.05)
    if i == 100:
        container.success("설치완료!!")


with st.spinner("설치 진행중", show_time=True):
    sleep(5)
st.success("설치 완료!!!")









with st.status("머신러닝 실행중", expanded=True) as status:
    st.write("데이터 입력중...")
    sleep(1)
    st.write("모델 학습중...")
    sleep(1)
    st.write("결과 도출중...")
    sleep(1)
    status.update(label="머신러닝 실행 완료", state="complete")


