from django.urls import path
from . import views


urlpatterns = [
    path('', views.index),
    path('var01/', views.variable01),
    path('var02/', views.variable02),
]