# 비어있는 문자열 : false
if "" :
    print("false")
else :
    print("true")

# [] {} () : false
if[] :
    print("false")
elif {} :
    print("false")
elif () :
    print("false")
elif [1] :
    print("true")

# 0 : false / 1 : true
if 0 :
    print("false")
elif 1 :
    print("true")

# None : false
if None :
    print("false")
else :
    print("true")