import seaborn as sns
import matplotlib.pyplot as plt


penguins = sns.load_dataset('penguins')
print(penguins)
print(penguins.columns)

"""
species : 종
island :  서식지
bill_length_mm : 부리의 길이
bill_depth_mm : 부리의 깊이
flipper_length_mm : 날개의 길이
body_mass_g : 체질량
sex : 성별
"""

# sns.boxplot(data=penguins, x='body_mass_g')
# sns.boxplot(data=penguins, x="bill_length_mm")
# sns.boxplot(data=penguins, y="bill_depth_mm")

# 각각의 종의 부리의 길이가 다르더라
# sns.boxplot(data=penguins, x='species', y='bill_depth_mm')

# hue : 색상을 맵핑하여 해당데이터를 나눠서 보여준다.
sns.boxplot(data=penguins, x='species', y='bill_depth_mm', hue='sex')


plt.show()




