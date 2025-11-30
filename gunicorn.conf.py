# Gunicorn configuration file for OCR service
import multiprocessing
import os

# Server socket
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:5001')
backlog = 2048

# Worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', 2))
worker_class = 'sync'
worker_connections = 1000
timeout = 60  # Longer timeout for OCR processing
keepalive = 2

# Process naming
proc_name = 'fint-ai-ocr'

# Logging
accesslog = os.environ.get('GUNICORN_ACCESS_LOG', '-')
errorlog = os.environ.get('GUNICORN_ERROR_LOG', '-')
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# Hooks
def on_starting(server):
    print("🤖 Fint AI OCR Service starting with Gunicorn...")

def on_exit(server):
    print("👋 Fint AI OCR Service shutting down...")
