import matplotlib.pyplot as plt

# 도화지 크기 만들기
fig = plt.figure(figsize=(10,5))
# ax = fig.subplots()

# 1행 2열로 만들고 몇번째에 위치 할거야
ax01=fig.add_subplot(1, 2, 1)
ax02=fig.add_subplot(1, 2, 2)
# fig, ax = pit.subplots()

x =[1, 2, 3, 4, 5]
y01 = list(map(lambda x: x**2, x))
y02 = list(map(lambda x: x**3, x))

ax01.plot([1, 2, 3, 4, 5],y01, color='red', label='pow2')
ax02.plot([1, 2, 3, 4, 5],y02, color='blue', label='pow3')


# 명칭 표시
# plt.legend()
ax01.legend()
ax02.legend()

ax01.set_title('x ** 2')
ax02.set_title('y ** 3')

plt.show()