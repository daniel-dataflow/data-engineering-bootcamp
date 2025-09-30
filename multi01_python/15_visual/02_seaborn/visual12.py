import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# 도화지 크기 그리기
fig = plt.figure(figsize=(10, 7))
# 하나의 행 2개의 열 중 왼쪽
ax01 = fig.add_subplot(1, 2, 1)
# 두번째 행 을 2개의 열 중 2번째
ax02 = fig.add_subplot(2, 2, 2)
ax03 = fig.add_subplot(2, 2, 4)

sns.histplot(data=penguins, x="body_mass_g", ax=ax01)
sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", ax=ax02)

# 결측치가 없으면 못그린다. 그래서 fillna로 데이터를 씌어준다.
# ax03.boxplot(penguins["body_mass_g"].fillna(penguins["body_mass_g"].mean()))
sns.boxplot(data=penguins, y="body_mass_g", ax=ax03)

plt.tight_layout()
plt.show()
