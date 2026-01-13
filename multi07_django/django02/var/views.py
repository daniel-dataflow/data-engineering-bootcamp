from django.shortcuts import render, redirect

def index(request):
    return render(request, 'var/index.html')

def variable01(request):
    my_list = ["python", "django", "template"]
    return render(request, 'var/variable01.html', {'lst': my_list})

def variable02(request):
    my_dict = {"class": "데이터 엔지니어", "name": "홍길동"}
    return render(request, 'var/variable02.html', {'dct': my_dict})

def for_loop(request):
    return render(request, 'var/forloop.html', {"number": range(1, 11)})

def if01(request):
    return render(request, 'var/if01.html', {'user': {'id': 'hong-gd', 'job': 'student'}})

def if02(request):
    return render(request, 'var/if02.html', {'role': 'manager', 'id': 'hong-gd'})

def href(request):
    return render(request, 'var/href.html')

def get_post(request):
    if request.method == "GET":
        return render(request, 'var/get.html')
    elif request.method == "POST":
        return render(request, 'var/post.html')
    else:
        return redirect('index')