from django.urls import path, include
from . import views


urlpatterns = [
    path("index/", views.index, name='index'),
    path("list/", views.list, name='list'),
    path("predict/", views.predict, name='predict'),
    path("info/", views.info, name="info"),
    path("predict/", views.predict, name='predict'),
    path("search_stocks/", views.search_stocks, name='search_stocks'),
    path('flag/<str:country_code>/', views.get_flag_image, name='flag_image'),
    path("sector_detail/<str:sector>/", views.sector_detail, name='sector_detail'),
]