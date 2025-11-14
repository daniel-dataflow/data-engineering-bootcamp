#5. 입력을 받아서 전체 학생 중 최고 점수를 받은 학생명과 점수를 출력하시오.
v_student = {}
while True:
   v_key = input('학생명을 입력하시오.(종료:q):')
   if v_key == 'q':
      break
   v_value = input(f'{v_key}의 점수를 입력하시오.:')
   v_student[v_key] = v_value
v_top_score = max(int(value) for value in v_student.values())
v_student_name = ''
for v_key,v_value in v_student.items():
   if v_value == str(v_top_score):
      v_student_name = v_key
      break
print(f'최고 점수를 받은 학생명은 {v_student_name}이고 최고 점수는 {v_top_score}')