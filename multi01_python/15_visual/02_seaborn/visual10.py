import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset('penguins')

# bill_로 시작하는 컬럼을 필터해줌(from pandas)
# ecdfplot :  경험적 누적분포함수
# 반복된 시행을 통해 확률 뼌수가 일정 값을 넘지 않을 확률을 유추하는 함수
sns.ecdfplot(data=penguins.filter(like="bill_", axis="columns"))


plt.show()




