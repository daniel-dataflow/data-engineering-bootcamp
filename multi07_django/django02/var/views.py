from django.shortcuts import render

def index(request):
    return render(request, 'var/index.html')

def variable01(request):
    my_list = ["python", "django", "template"]
    return render(request, 'var/variable01.html', {'lst': my_list})

def variable02(request):
    my_dict = {"class": "데이터 엔지니어", "name": "홍길동"}
    return render(request, 'var/variable02.html', {'dct': my_dict})




