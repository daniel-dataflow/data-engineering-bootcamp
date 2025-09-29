import matplotlib.pyplot as plt

# 도화지
fig = plt.figure()

# subplots : 도화지에 몇개로 자를건지
ax = fig.subplots()

# plot 그래프 그리기
ax.plot([1, 2, 3, 4 ,5])

# 보여줘
plt.show()