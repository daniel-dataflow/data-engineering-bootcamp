# str :  text sequence

# single * 1
a = 'hello, world'
print(a)
print(type(a))

# escape sequence
b = 'hello, \'python\''
print(b)

# single * 3
c = '''hello
python 
     abc
        def'''
print(c)

# double * 1
e = "hello, world"
print(e)

# double * 3
f = """hello 
python
    abc
       def
"""
print(f)


# 혼합
g = "hello, 'pthon'"
print(g)
h ='hello, "python"'
print(h)

# str()
i = str("hello, world")
print(i)

# escape sequence
print("hello, \nwolrd")

# raw sting
j = "c:\test"
print(j)
k = r"c:\test"
print(k)

# str  + str
print("hello" +  " " + "world" + "!!")
# print("hello" - "o")
print("hello" * 2)
# print("hello" + 1)
print("hello" + str(1))
