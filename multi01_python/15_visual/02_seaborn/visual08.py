import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# 어느 곳에 많이 분포되어 있는지 알고 싶을 때 사용함
# sns.violinplot(data=penguins, x="body_mass_g")
# sns.violinplot(data=penguins, x="body_mass_g", y="species")
# sns.violinplot(data=penguins, x="body_mass_g", y="species", hue='sex')
sns.violinplot(data=penguins, x="body_mass_g", y="species", hue='sex', split=True)


plt.show()