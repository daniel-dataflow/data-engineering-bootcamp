import streamlit as st
import pandas as pd


@st.cache_data
def load_csv():
    all_csv = pd.DataFrame()

    for i in range(19, 22):
        for j in range(1, 13):
            if j <10:
                j = f"0{j}"
            all_csv = pd.concat([all_csv, pd.read_csv(f"data/card_data/csv/20{i}{j}.csv")], ignore_index=True)
    return all_csv

all_data = load_csv()
st.session_state.all_data = all_data

pages = [st.Page("category.py", title="분류별"),
         st.Page("year.py", title="년도별"),
         st.Page("month.py", title="월별")
]

navi = st.navigation(pages)
navi.run()








