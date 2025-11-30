"""
OCR URL routes
"""
from django.urls import path
from . import views

urlpatterns = [
    path('scan', views.scan_receipt, name='scan_receipt'),
    path('provider', views.get_provider_info, name='provider_info'),
]
