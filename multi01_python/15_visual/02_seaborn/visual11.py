import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# 각 컬럼들을 가지고 스케터를 그리고 있고 같은 라인은 빈도를 나타내준다. 다른 컬럼간에 상관관계를 볼 수 있다.(양의 상관관계, 음의 상관관계)
# 같은 컬럼 => hist
# sns.pairplot(penguins)
# 같은 컬럼 => KDE로 나타나고 있다.
# sns.pairplot(penguins, hue='sex')

# kind : scatter, kde, hist, reg
# sns.pairplot(penguins, kind='reg')
# corner : 반대쪽의 날리고 싶으면
sns.pairplot(penguins, kind='reg', corner=True)




plt.show()