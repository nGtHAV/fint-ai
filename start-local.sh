#!/bin/bash

# ============================================================
# Fint AI OCR Service - Local Development Script
# ============================================================
# Run this script on your LOCAL machine with GPU to start
# the Surya OCR service for receipt scanning.
#
# Usage: ./start-local.sh [options]
#   Options:
#     --dev       Development mode with auto-reload (default)
#     --prod      Production mode with gunicorn
#     --port      Specify port (default: 5001)
#     --stop      Stop running service
#     --status    Show service status
#     --test      Test OCR with sample image
#     --help      Show this help message
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PORT="${PORT:-5001}"
MODE="dev"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Setup Virtual Environment
# =============================================================================

setup_venv() {
    print_step "Checking virtual environment..."
    
    if [[ ! -d "$VENV_DIR" ]]; then
        print_step "Creating virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate venv
    source "$VENV_DIR/bin/activate"
    
    # Check if surya-ocr is installed
    if ! python -c "import surya" 2>/dev/null; then
        print_step "Installing dependencies (this may take a while)..."
        pip install --upgrade pip
        pip install -r "$SCRIPT_DIR/requirements.txt"
    fi
    
    print_success "Virtual environment ready"
}

# =============================================================================
# Check GPU
# =============================================================================

check_gpu() {
    print_step "Checking GPU availability..."
    
    python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: No GPU detected - OCR will be slower")
PYEOF
    echo ""
}

# =============================================================================
# Create .env if not exists
# =============================================================================

setup_env() {
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        print_step "Creating .env file..."
        
        SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")
        
        cat > "$SCRIPT_DIR/.env" << EOF
# Django Settings
SECRET_KEY=${SECRET_KEY}
DEBUG=True

# OCR Configuration - Surya OCR (GPU optimized, ~2-3GB VRAM)
AI_PROVIDER=surya

# Allowed hosts
ALLOWED_HOSTS=localhost,127.0.0.1

# CORS - Allow frontend to access
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
EOF
        print_success "Created .env file"
    fi
}

# =============================================================================
# Start Development Server
# =============================================================================

start_dev() {
    print_header "Starting Fint AI (Development Mode)"
    
    cd "$SCRIPT_DIR"
    setup_venv
    setup_env
    check_gpu
    
    # Kill any existing process on the port
    print_step "Checking port $PORT..."
    lsof -ti:$PORT | xargs -r kill -9 2>/dev/null || true
    sleep 1
    
    print_success "Starting Django development server on port $PORT..."
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Fint AI OCR Service (Surya OCR + GPU)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "  ${BLUE}Local:${NC}    http://localhost:$PORT"
    echo -e "  ${BLUE}Network:${NC}  http://$(hostname -I | awk '{print $1}'):$PORT"
    echo ""
    echo -e "  ${BLUE}API Endpoints:${NC}"
    echo -e "    POST /api/ocr/scan    - Scan receipt image"
    echo -e "    GET  /api/ocr/info    - Provider information"
    echo ""
    echo -e "  ${YELLOW}Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    python manage.py runserver 0.0.0.0:$PORT
}

# =============================================================================
# Start Production Server
# =============================================================================

start_prod() {
    print_header "Starting Fint AI (Production Mode)"
    
    cd "$SCRIPT_DIR"
    setup_venv
    setup_env
    check_gpu
    
    # Kill any existing process on the port
    print_step "Checking port $PORT..."
    lsof -ti:$PORT | xargs -r kill -9 2>/dev/null || true
    sleep 1
    
    print_success "Starting Gunicorn on port $PORT..."
    echo ""
    
    # Update gunicorn config for the port
    export PORT=$PORT
    
    gunicorn --config gunicorn.conf.py fint_ai.wsgi:application
}

# =============================================================================
# Start as Background Service
# =============================================================================

start_background() {
    print_header "Starting Fint AI (Background Mode)"
    
    cd "$SCRIPT_DIR"
    setup_venv
    setup_env
    
    # Kill any existing process
    lsof -ti:$PORT | xargs -r kill -9 2>/dev/null || true
    sleep 1
    
    print_step "Starting in background..."
    
    nohup "$VENV_DIR/bin/python" manage.py runserver 0.0.0.0:$PORT > /tmp/fint-ai.log 2>&1 &
    
    sleep 2
    
    if lsof -ti:$PORT > /dev/null 2>&1; then
        print_success "Service started on port $PORT"
        print_info "View logs: tail -f /tmp/fint-ai.log"
    else
        print_error "Failed to start service"
        cat /tmp/fint-ai.log
        exit 1
    fi
}

# =============================================================================
# Stop Service
# =============================================================================

stop_service() {
    print_header "Stopping Fint AI Service"
    
    if lsof -ti:$PORT > /dev/null 2>&1; then
        print_step "Stopping service on port $PORT..."
        lsof -ti:$PORT | xargs -r kill -9
        sleep 1
        print_success "Service stopped"
    else
        print_info "No service running on port $PORT"
    fi
}

# =============================================================================
# Show Status
# =============================================================================

show_status() {
    print_header "Fint AI Service Status"
    
    echo -e "${CYAN}Port $PORT:${NC}"
    if lsof -ti:$PORT > /dev/null 2>&1; then
        PID=$(lsof -ti:$PORT | head -1)
        echo -e "  ${GREEN}● Running${NC} (PID: $PID)"
        
        # Test health endpoint
        echo ""
        echo -e "${CYAN}Health Check:${NC}"
        if curl -s "http://localhost:$PORT/api/ocr/info/" > /dev/null 2>&1; then
            echo -e "  ${GREEN}● API responding${NC}"
            curl -s "http://localhost:$PORT/api/ocr/info/" | python3 -m json.tool 2>/dev/null || true
        else
            echo -e "  ${YELLOW}● API not responding yet (may be loading models)${NC}"
        fi
    else
        echo -e "  ${RED}● Not running${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}GPU Status:${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    else
        echo "  nvidia-smi not available"
    fi
}

# =============================================================================
# Test OCR
# =============================================================================

test_ocr() {
    print_header "Testing Surya OCR"
    
    cd "$SCRIPT_DIR"
    setup_venv
    
    print_step "Running OCR test..."
    
    python3 << 'PYEOF'
from PIL import Image, ImageDraw
import io
import time

# Create test receipt image
print("Creating test receipt image...")
img = Image.new("RGB", (400, 300), color="white")
draw = ImageDraw.Draw(img)
draw.text((50, 20), "COFFEE SHOP", fill="black")
draw.text((50, 50), "123 Main Street", fill="black")
draw.text((50, 90), "Latte              $5.50", fill="black")
draw.text((50, 120), "Croissant          $3.00", fill="black")
draw.text((50, 160), "------------------------", fill="black")
draw.text((50, 190), "Total:             $8.50", fill="black")
draw.text((50, 230), "Date: 2024-12-01", fill="black")
draw.text((50, 260), "Thank you!", fill="black")

buffer = io.BytesIO()
img.save(buffer, format="PNG")
image_data = buffer.getvalue()

# Test Surya OCR
print("\nInitializing Surya OCR...")
start = time.time()

from apps.ocr.services import SuryaOCR_Provider
provider = SuryaOCR_Provider()

init_time = time.time() - start
print(f"Initialization time: {init_time:.2f}s")

print("\nProcessing image...")
start = time.time()
result = provider.extract_receipt_data(image_data)
process_time = time.time() - start

print(f"Processing time: {process_time:.2f}s")
print("\n" + "="*50)
print("RESULT:")
print("="*50)
import json
print(json.dumps(result, indent=2, default=str))
PYEOF
}

# =============================================================================
# Show Help
# =============================================================================

show_help() {
    echo ""
    echo -e "${PURPLE}Fint AI OCR Service - Local Development${NC}"
    echo ""
    echo "Usage: ./start-local.sh [options]"
    echo ""
    echo "Options:"
    echo "  --dev         Start in development mode with auto-reload (default)"
    echo "  --prod        Start in production mode with gunicorn"
    echo "  --bg          Start in background mode"
    echo "  --port PORT   Specify port (default: 5001)"
    echo "  --stop        Stop running service"
    echo "  --status      Show service status"
    echo "  --test        Test OCR with sample image"
    echo "  --gpu         Check GPU status"
    echo "  --help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./start-local.sh                  # Start dev server on port 5001"
    echo "  ./start-local.sh --port 8000      # Start on port 8000"
    echo "  ./start-local.sh --bg             # Start in background"
    echo "  ./start-local.sh --test           # Test OCR functionality"
    echo "  ./start-local.sh --stop           # Stop the service"
    echo ""
    echo "Environment Variables:"
    echo "  PORT=5001     Override default port"
    echo ""
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                MODE="dev"
                shift
                ;;
            --prod)
                MODE="prod"
                shift
                ;;
            --bg|--background)
                MODE="bg"
                shift
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --stop)
                stop_service
                exit 0
                ;;
            --status)
                show_status
                exit 0
                ;;
            --test)
                test_ocr
                exit 0
                ;;
            --gpu)
                setup_venv
                check_gpu
                exit 0
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Run based on mode
    case $MODE in
        dev)
            start_dev
            ;;
        prod)
            start_prod
            ;;
        bg)
            start_background
            ;;
    esac
}

# Run
main "$@"
