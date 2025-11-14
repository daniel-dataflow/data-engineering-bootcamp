v_number = [2, 43, 1, 6, 94, 76, 193, 0]
v_even = 0
v_odd = 0
for v_count in v_number:
   if (v_count % 2) == 0:
      v_even += 1
   else:
      v_odd += 1
print(f'홀수 : {v_odd}\n짝수 : {v_even}')