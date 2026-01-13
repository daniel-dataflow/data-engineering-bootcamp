import streamlit as st
import pandas as pd
import pydeck as pdk
import pickle


DATA_PATH = "data/processed/result.csv"
MODEL_PATH = "data/processed/model.pkl"
FEATURES = ["avg_day_noise","avg_night_noise","wgs84Lat","wgs84Lon"]

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df = load_data()
clf = load_model()

st.sidebar.title("🔍 필터")

noise_type = st.sidebar.radio(
    "소음 유형 선택",
    ("주간 소음", "야간 소음")
)

show_only_problem = st.sidebar.checkbox(
    "개선 필요 지역만 보기",
    value=False
)

st.sidebar.subheader("ML 위치 예측")
input_day = st.sidebar.number_input("주간 소음(dB)", 0.0, 150.0, 60.0)
input_night = st.sidebar.number_input("야간 소음(dB)", 0.0, 150.0, 55.0)
input_lat = st.sidebar.number_input("위도", value=float(df["wgs84Lat"].mean()))
input_lon = st.sidebar.number_input("경도", value=float(df["wgs84Lon"].mean()))

radius_meter = st.sidebar.slider("영향 반경 (m)", 100, 2000, 500, step=100)

if noise_type == "주간 소음":
    noise_col = "avg_day_noise"
    exceed_col = "day_exceeded"
    threshold = 65
else:
    noise_col = "avg_night_noise"
    exceed_col = "night_exceeded"
    threshold = 55

view_df = df.copy()
if show_only_problem:
    view_df = view_df[view_df["need_improvement"] == 1]

st.title("도시 소음 현황 대시보드")

col1, col2, col3 = st.columns(3)
col1.metric("전체 측정 지점 수", len(df))
col2.metric("기준 초과 지점 수", int(df[exceed_col].sum()))
col3.metric("기준 소음 (dB)", threshold)

st.subheader("소음 분포 지도")

view_df["color"] = view_df[exceed_col].apply(
    lambda x: [255, 0, 0] if x == 1 else [0, 128, 255]
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=view_df,
    get_position=["wgs84Lon", "wgs84Lat"],
    get_radius=80,
    get_fill_color="color",
    pickable=True,
)

input_df = pd.DataFrame([{
    "name": "입력 위치",
    "lon": input_lon,
    "lat": input_lat
}])

radius_layer = pdk.Layer(
    "ScatterplotLayer",
    data=input_df,
    get_position=["lon", "lat"],
    get_radius=radius_meter,
    get_fill_color=[255, 215, 0, 60],  # 노란색 + 투명도
    get_line_color=[255, 215, 0],
    stroked=True,
    filled=True,
    pickable=False,
)


view_state = pdk.ViewState(
    latitude=input_lat,
    longitude=input_lon,
    zoom=12,
)

st.pydeck_chart(
    pdk.Deck(
        layers=[point_layer, radius_layer],
        initial_view_state=view_state,
        tooltip={
            "html": """
                <b>{spotName}</b><br/>
                주소: {spotAddr}<br/>
                소음(dB): {""" + noise_col + """}
            """
        }
    )
)

st.subheader("상세 데이터")
st.dataframe(
    view_df[
        ["spotName","spotAddr",noise_col,exceed_col,"need_improvement"]
    ].rename(columns={
        noise_col: "평균 소음(dB)",
        exceed_col: "기준 초과 여부",
        "need_improvement": "개선 필요"
    })
)

st.subheader("입력 위치 개선 필요 여부 예측")

new_data = pd.DataFrame([{
    "avg_day_noise": input_day,
    "avg_night_noise": input_night,
    "wgs84Lat": input_lat,
    "wgs84Lon": input_lon
}])

pred = clf.predict(new_data)[0]
prob = clf.predict_proba(new_data)[0][1]

if pred == 1:
    st.error(f"이 위치는 **개선 필요** 예상 (확률: {prob:.2f})")
else:
    st.success(f"이 위치는 **개선 필요 없음** 예상 (확률: {prob:.2f})")
