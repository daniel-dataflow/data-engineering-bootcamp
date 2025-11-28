import streamlit as st

st.title("home")
for i in range(10):
    st.html("<br>")

st.title("go home", anchor="home")

# 줄 그리기
st.divider()

for i in range(11):
    st.header(i, divider=True)

st.subheader("sub header")
st.subheader("sub header", divider=True)

st.subheader("sub header", width="stretch", divider=True)
st.subheader("sub header", width=200, divider=True)

markdown = """
# 마크다운
## 문법은
### 매우
#### 중요합니다.
"""
st.markdown(markdown)

st.badge("streamlit")
st.badge("streamlit", color="green")
st.markdown(":orange-badge[warning] :red-badge[error]")

st.divider()

st.caption("이건 **설명** 입니다.")
st.caption("이건 ```설명``` 입니다.")

code ="""
import streamlit as st

st.header("home")
st.text("우리집 주소는 서울시 도봉구 창동 ...."
"""
st.code(code)
st.markdown(f"```{code}")
st.markdown(f"```python {code}")

with st.echo():
    for i in range(10):
        st.text("!!")

# latex : 수학식을 표현하기 위해 만들어진 문법
st.latex(r"""
    \dot{a}, \ddot{a}, \acute{a}, \grave{a}
""")


