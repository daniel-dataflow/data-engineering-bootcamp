# parameter :  함수 외부에서 전달되는 값을 받아서, 함수 **내부**에서 사용하기 위한 "변수"(매개변수)
from yt_dlp.extractor.podbayfm import result_from_props


def sum2(x):
    return x + 2

# argument : 함수 **외부**에서 전달되는 "값"
print(sum2(5))

def sum_x_y(x, y):
    return x + y
    return result

print(sum_x_y(2, 3))

print(sum_x_y(x=10, y=20))
print(sum_x_y(y=30, x=50))


print(f"__name__ : {__name__}")
# __name__ : 내가 직접 실행돠면 "__main__" 을 가진다.
#            만일, 외부에서 호출되면 파일의 이름을 가진다.

if __name__ == "__main__":
    print(sum2(100))
    print(sum_x_y(1000, 2000))
    
