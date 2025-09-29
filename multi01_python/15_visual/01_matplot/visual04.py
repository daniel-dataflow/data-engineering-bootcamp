import matplotlib.pyplot as plt
import random

# fig = plt.figure()
# ax01=fig.add_subplot(2, 2, 1)
# ax02=fig.add_subplot(2, 2, 2)
# ax03=fig.add_subplot(2, 2, 3)
# ax04=fig.add_subplot(2, 2, 4)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

x = list(range(50))
y01 = list(random.randint(0 , 50) for _ in range(50))
y02 = list(random.randint(0 , 50) for _ in range(50))
y03 = list(random.randint(0 , 50) for _ in range(50))
y04 = list(random.randint(0 , 50) for _ in range(50))

# scatter (산포도) : 데이터의 분포 확인. x와 y의 관계 파악
ax[0, 0].scatter(x, y01, color = 'red')
ax[0, 1].scatter(x, y02, color = 'green')
ax[1, 0].scatter(x, y03, color = 'blue')
ax[1, 1].scatter(x, y04, color = 'yellow')

# 자동간격
ax[0, 0].set_title('y01')
ax[0, 1].set_title('y02')
ax[1, 0].set_title('y03')
ax[1, 1].set_title('y04')

# 자동으로 레이아웃 설정
fig.tight_layout()

plt.show()

