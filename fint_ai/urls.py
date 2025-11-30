"""
URL configuration for fint_ai OCR service.
"""
from django.urls import path, include
from django.http import JsonResponse
from datetime import datetime


def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'service': 'fint-ai-ocr',
        'timestamp': datetime.now().isoformat()
    })


urlpatterns = [
    path('api/health', health_check, name='health_check'),
    path('api/ocr/', include('apps.ocr.urls')),
]
