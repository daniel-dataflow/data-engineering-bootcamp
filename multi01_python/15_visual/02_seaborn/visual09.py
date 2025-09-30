import seaborn as sns
import matplotlib.pyplot as plt


penguins = sns.load_dataset('penguins')

# regplot = scatter + line (추세선)
sns.regplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")



plt.show()

