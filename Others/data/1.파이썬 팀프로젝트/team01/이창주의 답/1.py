#1. 입력을 받아서 소수인지, 아닌지 판별 결과를 표현하시오.
def f_is_prime(a_number:int):
   if a_number < 2:
      return False
   for v_count in range(2,int(a_number ** 0.5) + 1):
      if a_number % v_count == 0:
         return False
   return True
v_number = input('원하는 숫자(소수인지 판별):')
if f_is_prime(int(v_number)):
   print('소수입니다.')
else:
   print('소수가 아닙니다.')