import matplotlib.pyplot as plt
from random import randint

fig, ax = plt.subplots()

ages = list(randint(1, 100) for i in range(100))

child = list( x for x in ages if x < 19)
young = list( x for x in ages if 19 <= x < 45)
middle =  list( x for x in ages if 45 <= x < 70)
old = list( x for x in ages if x >-70)

labels = ["child", "young", "middle", "old"]
count = [len(child), len(young), len(middle), len(old)]

# counterclock : 시계방향으로 나외라 startangle : 시작각도
ax.pie(count, labels=labels, autopct='%1.1f%%', counterclock=False, startangle=90)

plt.show()