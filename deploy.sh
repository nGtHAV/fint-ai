#!/bin/bash

# ============================================================
# Fint AI OCR Service - Remote Server Deployment Script
# ============================================================
# Deploy to a remote server (DigitalOcean, etc.)
# For LOCAL development with GPU, use: ./start-local.sh
#
# Usage: 
#   curl -sSL https://raw.githubusercontent.com/nGtHAV/fint-ai/main/deploy.sh | bash
#   Or: ./deploy.sh
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
APP_NAME="fint-ai"
APP_USER="fint"
APP_DIR="/opt/fint/ai"
REPO_URL="${REPO_URL:-https://github.com/nGtHAV/fint-ai.git}"
DOMAIN="${DOMAIN:-}"
PORT=5001

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       Fint AI OCR Service - Remote Deployment${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo -e "${YELLOW}Note: For local development with GPU, use ./start-local.sh${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo)${NC}"
    exit 1
fi

# Step 1: Update system
echo -e "${YELLOW}[1/8] Updating system packages...${NC}"
apt update && apt upgrade -y

# Step 2: Install dependencies
echo -e "${YELLOW}[2/8] Installing system dependencies...${NC}"
apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    nginx \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libglx-mesa0

# Step 3: Create application user
echo -e "${YELLOW}[3/8] Setting up application user...${NC}"
if ! id "$APP_USER" &>/dev/null; then
    adduser --system --group --home /opt/fint $APP_USER
fi

# Step 4: Clone/Update repository
echo -e "${YELLOW}[4/8] Setting up application...${NC}"
mkdir -p /opt/fint
cd /opt/fint

if [ -d "ai" ]; then
    echo "Updating existing installation..."
    cd ai
    git pull origin main
else
    echo "Cloning repository..."
    git clone $REPO_URL ai
    cd ai
fi

# Step 5: Setup Python environment
echo -e "${YELLOW}[5/8] Setting up Python environment...${NC}"

# Prompt for OCR provider selection
echo ""
echo -e "${BLUE}Select OCR Provider:${NC}"
echo "1) EasyOCR - Lightweight, good accuracy, works on CPU/GPU (default)"
echo "2) Surya OCR - Best accuracy, optimized for GPU (~2-3GB VRAM)"
echo ""
read -p "Choice [1/2] (default: 1): " OCR_CHOICE
OCR_CHOICE=${OCR_CHOICE:-1}

case $OCR_CHOICE in
    2)
        OCR_PROVIDER="surya"
        echo -e "${GREEN}Using Surya OCR (GPU optimized)${NC}"
        ;;
    *)
        OCR_PROVIDER="easyocr"
        echo -e "${GREEN}Using EasyOCR (CPU/GPU compatible)${NC}"
        ;;
esac

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies based on selected provider
echo -e "${YELLOW}Installing dependencies for ${OCR_PROVIDER}...${NC}"
if [[ "$OCR_PROVIDER" == "easyocr" ]]; then
    pip install --no-cache-dir -r requirements-easyocr.txt
else
    pip install --no-cache-dir -r requirements-surya.txt
fi

# Step 6: Configure environment
echo -e "${YELLOW}[6/8] Configuring environment...${NC}"
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(50))")
cat > .env << EOF
SECRET_KEY=$SECRET_KEY
DEBUG=False
AI_PROVIDER=${OCR_PROVIDER}
ALLOWED_HOSTS=localhost,127.0.0.1${DOMAIN:+,$DOMAIN}
CORS_ALLOWED_ORIGINS=http://localhost:3000${DOMAIN:+,https://$DOMAIN}
EOF
echo -e "${GREEN}Created .env file with AI_PROVIDER=${OCR_PROVIDER}${NC}"

# Step 7: Setup systemd service
echo -e "${YELLOW}[7/8] Setting up systemd service...${NC}"
cat > /etc/systemd/system/$APP_NAME.service << EOF
[Unit]
Description=Fint AI OCR Service (${OCR_PROVIDER})
After=network.target

[Service]
Type=exec
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR
Environment="PATH=$APP_DIR/venv/bin"
ExecStart=$APP_DIR/venv/bin/gunicorn --config gunicorn.conf.py fint_ai.wsgi:application
Restart=always
RestartSec=3

# OCR may need more memory
MemoryMax=4G
CPUQuota=100%

[Install]
WantedBy=multi-user.target
EOF

# Step 8: Setup Nginx
echo -e "${YELLOW}[8/8] Setting up Nginx...${NC}"
cat > /etc/nginx/sites-available/$APP_NAME << EOF
server {
    listen 80;
    server_name ${DOMAIN:-_};

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Increase timeout for OCR processing
        proxy_read_timeout 120s;
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        
        # Increase body size for image uploads
        client_max_body_size 16M;
    }

    location /api/health {
        proxy_pass http://127.0.0.1:$PORT/api/health;
        proxy_set_header Host \$host;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/$APP_NAME /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Set ownership
chown -R $APP_USER:$APP_USER $APP_DIR

# Reload services
systemctl daemon-reload
systemctl enable $APP_NAME
systemctl restart $APP_NAME
systemctl reload nginx

# Wait for service to start
sleep 3

# Check status
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}       Deployment Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "Service Status:"
systemctl status $APP_NAME --no-pager -l || true
echo ""
echo -e "${BLUE}Quick Commands:${NC}"
echo -e "  View logs:     ${YELLOW}journalctl -u $APP_NAME -f${NC}"
echo -e "  Restart:       ${YELLOW}sudo systemctl restart $APP_NAME${NC}"
echo -e "  Stop:          ${YELLOW}sudo systemctl stop $APP_NAME${NC}"
echo ""
echo -e "${BLUE}API Endpoints:${NC}"
echo -e "  Health:        ${YELLOW}http://your-ip/api/health${NC}"
echo -e "  Scan Receipt:  ${YELLOW}POST http://your-ip/api/ocr/scan${NC}"
echo -e "  Provider Info: ${YELLOW}http://your-ip/api/ocr/provider${NC}"
echo ""

if [ -z "$DOMAIN" ]; then
    echo -e "${YELLOW}To add SSL, run:${NC}"
    echo -e "  sudo apt install certbot python3-certbot-nginx -y"
    echo -e "  sudo certbot --nginx -d your-domain.com"
fi

echo ""
echo -e "${GREEN}Done!${NC}"
