import streamlit as st
import os

from streamlit import navigation

page_list = [file for file in os.listdir() if file.endswith(".py")]
page_list = [file for file in page_list if not file.startswith("09")]

streamlit_page = list()
for page in page_list:
    streamlit_page.append(st.Page(page))

navigation = st.navigation(streamlit_page)
navigation.run()