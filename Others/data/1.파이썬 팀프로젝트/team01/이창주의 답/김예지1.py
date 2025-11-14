def f_count(a_i:int):
   if a_i == 0:
      return 9
   if a_i == 1:
      return 5
   if a_i == 2:
      return 1
   if a_i == 3:
      return -3
   if a_i == 4:
      return -7
print('\n'.join([''.join([' ' for v_i in range(v_i)] + ['*' for v_i in range(2 * v_i + f_count(v_i))]) for v_i in range(5)]))