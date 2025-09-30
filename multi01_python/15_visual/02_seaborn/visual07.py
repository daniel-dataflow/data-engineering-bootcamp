import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
# sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue='sex')
# sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue='species', size='sex')
# sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue='species', style='sex')

markers = {"Male" : "P", "Female" : "o"}
sns.scatterplot(data=penguins, x="bill_length_mm", y="bill_depth_mm", hue='species', style='sex', markers=markers)

plt.show()