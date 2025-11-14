v_aliquot = int(input('원하는 약수:'))
v_result = lambda v_aliquot:list(filter(lambda v_x:v_aliquot % v_x == 0,range(1,v_aliquot + 1)))
print(v_result(v_aliquot))