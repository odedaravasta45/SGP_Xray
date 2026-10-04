from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_view, name='upload'),
    path('analyze/', views.analyze_view, name='analyze'),
    path('result/<str:result_id>/', views.result_view, name='result'),
]
