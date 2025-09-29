import matplotlib.pyplot as plt
from random import randint

fig, ax = plt.subplots()

x = list(randint(0, 10) for _ in range(100))

# hist :?
# bins :  막대의 갯수
ax.hist(x, bins=10)

# 축 눈금
plt.xticks(list(range(0, 11)))
plt.yticks(list(range(0, 101, 5)))

# 축 제한
plt.xlim(0, 10)
plt.ylim(0 , 100)

plt.show()