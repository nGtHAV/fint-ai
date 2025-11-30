# Fint AI OCR Service

A Django-based microservice for scanning receipts using **Surya OCR** with GPU acceleration. Optimized for 6GB VRAM GPUs.

## 🚀 Features

- **Surya OCR (Default)**: Modern, efficient OCR optimized for documents (~2-3GB VRAM)
- **GPU Accelerated**: Uses CUDA for faster processing
- **Multiple Providers**: Surya, EasyOCR, Tesseract, Google Gemini, OpenAI GPT-4 Vision
- **Smart Receipt Parsing**: Automatically extracts merchant name, total amount, date, and category
- **Line Item Detection**: Identifies individual items and their prices
- **REST API**: Simple JSON API for seamless integration

## 📋 Prerequisites

- Python 3.10+
- 4GB+ RAM (8GB+ recommended)
- NVIDIA GPU with 6GB+ VRAM (for Surya OCR)
- CUDA 12.x
- pip (Python package manager)

## 🛠️ Quick Start (Local Development)

### Option 1: Use the Start Script (Recommended)

```bash
# Start the AI service locally
./start-local.sh

# Or with options
./start-local.sh --port 5001     # Custom port
./start-local.sh --test          # Test OCR
./start-local.sh --status        # Check status
./start-local.sh --stop          # Stop service
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS

# Install dependencies (this takes a while for PyTorch/Surya)
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(50))")
DEBUG=True
AI_PROVIDER=surya
EOF

# Run the server
python manage.py runserver 0.0.0.0:5001
```

## 📚 API Endpoints

### Scan Receipt

```http
POST /api/ocr/scan
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "language": "en"  // optional
}
```

**Success Response:**
```json
{
    "success": true,
    "merchant": "Coffee Shop",
    "total": 8.50,
    "date": "2024-12-01",
    "category": "Food & Dining",
    "items": [
        {"name": "Latte", "price": 5.50},
        {"name": "Croissant", "price": 3.00}
    ],
    "raw_text": "...",
    "language": "en",
    "provider": "surya"
}
```

### Get Provider Info

```http
GET /api/ocr/info
```

### Health Check

```http
GET /api/health
```

## 🤖 AI Provider Comparison

| Provider | Accuracy | Speed | VRAM | Cost |
|----------|----------|-------|------|------|
| **Surya OCR** | Very High | Fast | ~2-3GB | Free |
| **EasyOCR** | High | Fast | ~3-4GB | Free |
| **Tesseract** | Medium | Fast | 0 | Free |
| **Gemini** | Very High | Medium | 0 | Pay-per-use |
| **OpenAI** | Highest | Medium | 0 | Pay-per-use |

## 📊 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | Active OCR provider | `surya` |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `SECRET_KEY` | Django secret key | Auto-generated |
| `DEBUG` | Debug mode | `False` |

## �️ Running Locally with GPU

When deploying the main Fint app to a server without GPU, you can run the AI service locally on your GPU machine:

```bash
# On your local machine with GPU
cd fint-ai
./start-local.sh

# The service will be available at http://localhost:5001
# Configure your deployed frontend to use this URL
```

### Exposing Local AI via Cloudflare Tunnel

```bash
# Install cloudflared
# Then create a tunnel to your local AI service
cloudflared tunnel --url http://localhost:5001
```

---

## 🚀 Deployment

### Deploy to Remote Server (without GPU)

```bash
# On the server
./deploy.sh
```

### Deploy Full Stack (Main deploy script)

```bash
# Deploy everything including AI
./deploy.sh --all

# Deploy without AI (run AI locally)
./deploy.sh --no-ai
```

## 📄 License

MIT License
