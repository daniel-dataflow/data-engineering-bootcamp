from ctypes.macholib.dyld import dyld_image_suffix_search
from random import randint

# 생년월일과 성별을 입력받아 주민등록번호를 완성하세요
# 뒤의 6자리는 랜덤한 숫자를 부여한다
# random 모듈, **kwargs, format() 활용

"""
몇년도에 태어나셨습니까? 2001
몇월에 태어나셨습니까? 2
몇일에 태어났습니까? 13
성별이 무엇입니까(남/여)? 남
주민등록번호: 010213-3124579
"""

def resident_number(**kwargs) -> None:
    print(kwargs)
    year_cut_str = kwargs['year'][2:]
    year_int = int(kwargs['year'])
    end_num = str(randint(100000, 999999))
    print(end_num)

    if (year_int < 2000):
        if (kwargs["gender"]=="남" ):
            g = '1'
        else:
            g = '2'
    else:
        if (kwargs["gender"]=="남"):
            g = '3'
        else:
            g = '4'

    return print(f"주민등록번호 : {year_cut_str+kwargs['month']+kwargs['day']}-{g+end_num}")

if __name__ == "__main__":
    year = input("몇 년도에 태어나셨습니까? : ")
    month = input("몇 월에 태어나셨습니까? : ")
    day = input("몇 일에 태어나셨습니까? : ")
    gender = input("성별은 무엇입니까?(남/여) : ")


    resident_number(**{"year":year, "month": month, "day": day, "gender" :gender})
