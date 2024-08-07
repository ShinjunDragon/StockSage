from django.urls import path, include
from . import views


urlpatterns = [
    path("index/", views.index, name='index'),
    path("list/", views.list, name='list'),
    path("predict/", views.predict, name='predict'),
    path("info/", views.info, name="info"),
    path('delete_comment/<int:comment_id>/', views.delete_comment, name='delete_comment'),
    path("predict/", views.predict, name='predict'),
    path("search_stocks/", views.search_stocks, name='search_stocks'),
    path('flag/<str:country_code>/', views.get_flag_image, name='flag_image'),
    path("sector_detail/<str:sector>/", views.sector_detail, name='sector_detail'),
    path("industry_detail/<str:industry>/", views.industry_detail, name='industry_detail'),
    path("combined_view/", views.combined_view, name='combined_view'),
    path('toggle_interest_stock/', views.toggle_interest_stock, name='toggle_interest_stock'),
]