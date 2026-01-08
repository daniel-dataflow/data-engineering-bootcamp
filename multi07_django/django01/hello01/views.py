from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    return HttpResponse("<h1><a href='/hello01/test'>Hello, django test</a></h1>")

def test(request):
    return HttpResponse("<a href='/hello01'>returnt</a>")


