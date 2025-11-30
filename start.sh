#!/bin/bash

# Fint AI OCR Service Start Script
# Usage: ./start.sh [development|production]

set -e

MODE=${1:-development}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Set environment variables
if [ "$MODE" = "production" ]; then
    export DJANGO_SETTINGS_MODULE=fint_ai.settings
    echo "🤖 Starting Fint AI OCR Service in PRODUCTION mode..."
    
    # Run with Gunicorn
    gunicorn --config gunicorn.conf.py fint_ai.wsgi:application
else
    export DJANGO_SETTINGS_MODULE=fint_ai.settings
    export DEBUG=True
    echo "🤖 Starting Fint AI OCR Service in DEVELOPMENT mode..."
    
    # Run development server on port 5001
    python manage.py runserver 0.0.0.0:5001
fi
