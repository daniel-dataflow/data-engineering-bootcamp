subjects = ["python","numpy","pandas"]

# enumerate : iterable 한 객체에 index 추가해서 리턴
print(enumerate(subjects))
print(dict(enumerate(subjects)))
print(list(enumerate(subjects)))

for idx, item in enumerate(subjects):
    print(f"{idx} : {item}")

