# 반복문을 사용하여 1 ~ 100 까지 모두 더한 결과를 출력하자
sum_hundred = 0
for i in range(101):
    sum_hundred += i




print(f'1 ~ 100 까지의 결과 : {sum_hundred}')

# 반복문을 사용하여 1 ~ 100 까지 홀수합 / 짝수합을 각각 출력하자
sum_odd = 0
sum_even = 0
for i in range(101):
    if(i % 2 == 0):
        sum_even += i
    else:
        sum_odd += i




print(f'홀수합 : {sum_odd}')
print(f'짝수합 : {sum_even}')
