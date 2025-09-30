import plotly.express as px
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 2020년 기준 상위 3개국의 올림픽 쇼트트랙 스피드 스케이팅 메달
df = px.data.medals_long()
print(df)

# nation 값이 South Korea인 데이터만 가지고 만들고 싶다.
korea = df[df["nation"] == "South Korea"]
print(korea)


fig = px.pie(korea, values="count", names="medal")
fig.write_image("korea.png")