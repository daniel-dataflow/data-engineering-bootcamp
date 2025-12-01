import streamlit as st


with st.form("form01"):
    st.header("Form01")
    txt_form01 = st.text_input(label="txt_form01")
    if st.form_submit_button("submit 01"):
        st.write(f"input txt : {txt_form01}")
    else:
        st.write(f"input txt: None")

form02 = st.form("form02")
form02.header("form02")
txt_form02 = form02.text_input(label="txt_form02")
submit = form02.form_submit_button("submit 02")

if submit:
    st.write(f"txt_form02 (st) input : {txt_form02}")
    form02.write(f"txt_form02 (form02) input : {txt_form02}")



