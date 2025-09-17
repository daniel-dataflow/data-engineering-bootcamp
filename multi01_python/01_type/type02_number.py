# int: 정수
a = 100
print(a)

# int 생성자  - > 객체생성 (class 때 배울거야)
b =  int(101)
print(b)

# flost  : 실수
c = 200.2
print(c)

d = float(200.3)
print(d)

# int + float = float
print(a + d)
print(type(a+d))


# complex  :  복소수
# real  = imaginary 허수 (실수부 + 하수부j)
e = 1+ 2j
print(e)
print(type(e))


#complex(real, image)
f = complex(3, 4)
print(f)

# boal :  논리
g = True
h = False
print(g)
print(h)

print(type(g))

# c언어 ->true : 1 /  false : 0  => python에서도 동일
print( 1 == g)
print(0 != g)

# 진수변경
# 2진수 (0b)
binary = 0b1111
print(binary)
# 8진수 (0o)
octal = 0o077
print(octal)
# 16진수 (0x)
hexadecimal = 0xff
print(hexadecimal)

