from django.urls import path
from . import views

# <a href="{% url 'index' %}">을 실행하기 위해 name='index' 작성
urlpatterns = [
    path('', views.index, name='index'),
    path('var01/', views.variable01),
    path('var02/', views.variable02),
    path('forloop/', views.for_loop),
    path('if01/', views.if01),
    path('if02/', views.if02),
    path('href/', views.href),
    path('request/', views.get_post),
]