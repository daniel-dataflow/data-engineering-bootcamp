#6. 원하는 년도와 월을 입력을 받아서 Python에서 제공하는 기능으로 달력을 출력하시오.
import calendar
v_year = input('년도:')
v_month = input('월:')
print(calendar.month(int(v_year),int(v_month)))