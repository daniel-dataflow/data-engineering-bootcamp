import matplotlib.pyplot as plt


# fig = plt.figure()
# ax = fig.subplots()
fig, ax = plt.subplots()

x =[1,2,3,4,5]
y01 = list(map(lambda x: x**2, x))
y02 = list(map(lambda x: x**3, x))

ax.plot([1,2,3,4,5],y01, color='red', label='pow2')
ax.plot([1,2,3,4,5],y02, color='blue', label='pow3')


# 명칭 표시
plt.legend()
plt.show()