from django.shortcuts import render, redirect
from .models import Myboard
from django.utils import timezone


def index(request):
    return render(request, 'index.html', {"list": Myboard.objects.all()})

def detail(request, id):
    return render(request, 'detail.html', {"dto": Myboard.objects.get(id=id)})

def insert(request):
    if request.method == "GET":
        return render(request, 'insert.html')

    elif request.method == "POST":
        myname = request.POST['myname']
        mytitle = request.POST['mytitle']
        mycontent = request.POST['mycontent']

        result = Myboard.objects.create(myname=myname, mytitle=mytitle, mycontent=mycontent, mydate=timezone.now())

        if result:
            return redirect("index")
        else:
            return redirect("insert")

def update(request, id):
    if request.method == "GET":
        return render(request, 'update.html', {"dto": Myboard.objects.get(id=id)})

    elif request.method == "POST":
        mytitle = request.POST['mytitle']
        mycontent = request.POST['mycontent']

        myboard = Myboard.objects.filter(id=id)
        result_title = myboard.update(mytitle=mytitle)
        result_content = myboard.update(mycontent=mycontent)

        if result_title + result_content == 2:
            return redirect(f"/detail/{id}")
        else:
            return redirect(f"/update/{id}")


def delete(request, id):
    result_delete = Myboard.objects.filter(id=id).delete()

    if result_delete[0]:
        return redirect("index")
    else:
        return redirect(f"detail/{id}")

