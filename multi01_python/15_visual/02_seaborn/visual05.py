import seaborn as sns
import matplotlib.pyplot as plt


penguins = sns.load_dataset('penguins')

# sns.histplot(penguins)
# sns.histplot(data=penguins, x='body_mass_g')
# sns.histplot(data=penguins, y='body_mass_g')
# sns.histplot(data=penguins, x='body_mass_g', bins=50)

# KDE :  Kernel Density Estimator : 커널 밀도 추정
# histogram은 bin의 크기에 따라 결과값이 다르게 보일 수 있다. => 커널함수를 사용하여 가질 수 있는 값을 확률로 보여줌
# sns.histplot(data=penguins, x='body_mass_g', binwidth=100, kde=True)

# multiple : layer, dodge, stack, fill
# sns.histplot(data=penguins, x='body_mass_g', hue='species', multiple="layer")
# sns.histplot(data=penguins, x='body_mass_g', hue='species', multiple="dodge")
# sns.histplot(data=penguins, x='body_mass_g', hue='species', multiple="stack")
# sns.histplot(data=penguins, x='body_mass_g', hue='species', multiple="fill")

sns.histplot(data=penguins, x='body_mass_g', hue='species', multiple="fill", fill=False)


plt.show()
