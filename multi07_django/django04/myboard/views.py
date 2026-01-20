from django.shortcuts import render, redirect
from .models import Myboard, MyMember
from django.utils import timezone
from django.core.paginator import Paginator
from django.contrib.auth.hashers import make_password, check_password


def index(request):
    # return render(request, 'index.html', {"list": Myboard.objects.all()})

    # -id : 최신글이 가장 위로 올라오게
    myboards = Myboard.objects.all().order_by('-id')
    paginator = Paginator(myboards, 5)
    # 만약 여러개의 페이지가 없으면 1로 셋팅된다.
    page_num = request.GET.get('page', '1')
    page_obj = paginator.get_page(page_num)

    ### 화면에서 나올 내용을 미리 확인해보자 ###
    # 타입은 paginator.Page 이다
    print(type(page_obj))
    # 글의 총 갯수
    print(page_obj.count)
    # 5개씩 나눴을 때의 페이지의 갯수
    print(page_obj.paginator.num_pages)
    # 그 페이지의 객체의 범위
    print(page_obj.paginator.page_range)
    # 다음 페이지 있는지
    print(page_obj.has_next())
    # 이전 페이지가 있는지
    print(page_obj.has_previous())

    # 이전 페이지가 없으면 에러가 나는 것을 방지
    try:
        print(page_obj.next_page_number())
        print(page_obj.previous_page_number())
    except:
        pass
    print(page_obj.start_index())
    print(page_obj.end_index())
    ###

    return render(request, 'index.html', {'list': page_obj})

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

def register(request):
    if request.method == "GET":
        return render(request, 'register.html')
    elif request.method == "POST":
        myname = request.POST['myname']
        mypassword = request.POST['mypassword']
        myemail = request.POST['myemail']

        mymember = MyMember.objects.create(myname=myname, mypassword=make_password(mypassword), myemail=myemail)
        mymember.save()
        return redirect("/")

def login(request):
    if request.method == "GET":
        return render(request, 'login.html')

    elif request.method == "POST":
        myname = request.POST['myname']
        mypassword = request.POST['mypassword']

        mymember = MyMember.objects.get(myname=myname)

        if check_password(mypassword, mymember.mypassword):
            request.session['myname'] = mymember.myname
            return redirect("/")
        else:
            return redirect("login")

def logout(request):
    del request.session['myname']
    return redirect("/")