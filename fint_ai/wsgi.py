"""
WSGI config for fint_ai OCR service.
"""
import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fint_ai.settings')

application = get_wsgi_application()
