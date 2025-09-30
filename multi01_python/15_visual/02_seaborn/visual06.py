import seaborn as sns
import matplotlib.pyplot as plt


penguins = sns.load_dataset('penguins')

# sns.swarmplot(data=penguins, x='body_mass_g')
# sns.swarmplot(data=penguins, x='body_mass_g', y="species")
sns.swarmplot(data=penguins, x='body_mass_g', y="species", color='black', alpha=0.5)
sns.boxplot(data=penguins, x='body_mass_g', y="species")


plt.show()
