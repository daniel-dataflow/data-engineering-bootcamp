from django.shortcuts import render, redirect
from .models import MyBoard
from django.utils import timezone

def index(request):
    return render(request, 'index.html', {"list": MyBoard.objects.all()})

def detail(request, id):
    return render(request, 'detail.html', {"dto": MyBoard.objects.get(id=id)})

def insert(request):
    if request.method == "GET":
        return render(request, "insert.html")

    elif request.method == "POST":
        myname = request.POST["myname"]
        mytitle = request.POST["mytitle"]
        mycontent = request.POST["mycontent"]

        result = MyBoard.objects.create(myname=myname, mytitle=mytitle, mycontent=mycontent, mydate=timezone.now())

        if result:
            return redirect("index")
        else:
            return redirect("insert")
    else:
        return redirect("index")


def update(request, id):
    if request.method == "GET":
        return render(request, "update.html", {"dto": MyBoard.objects.get(id=id)})

    elif request.method == "POST":
        mytitle = request.POST["mytitle"]
        mycontent = request.POST["mycontent"]

        mybored = MyBoard.objects.filter(id=id)
        result_title = mybored.update(mytitle=mytitle)
        result_content = mybored.update(mycontent=mycontent)

        if result_title + result_content ==2:
            return redirect(f"/detail/{id}")
        else:
            return redirect(f"/update/{id}")
    else:
        return redirect("index")


def delete(request, id):
    result_delete = MyBoard.objects.filter(id=id).delete()

    if result_delete[0]:
        return redirect("index")
    else:
        return redirect(f"/detail/{id}")
