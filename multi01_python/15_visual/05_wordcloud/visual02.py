from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

with open("webtoons.json", "r", encoding="utf-8") as file:
    webtoons = json.load(file)

res = dict()
for webtoon in webtoons["webtoons"]:
    res[webtoon["title"]] = int(float(webtoon["star"]) * 100) # 점수가 높은애들을 큰 글씨로 표현하게 만듬

masking_img = np.array(Image.open("kakao.png")) # 이미지를 n차원의 베열로 만들어버린다. - 이미지를 데이터러 바꿔서 사용 할 수 있다.
cloud = WordCloud(font_path="Goyang.otf", max_font_size=40,
                  mask=masking_img, background_color="white").fit_words(res)
cloud.to_file("cloud02.png")

plt.imshow(cloud, interpolation="bilinear") # 두사이의 경계를 잘 모를때 평균값으로 채우겠다.
plt.axis("off")
plt.show()