# Fint AI OCR Service

A Django-based microservice for scanning receipts using PaddleOCR with **English and Khmer (ភាសាខ្មែរ)** language support.

## 🚀 Features

- **PaddleOCR (Default)**: High-accuracy local OCR with no API costs
- **Multi-language**: English (en) and Khmer (km) support
- **Auto Language Fallback**: Automatically tries alternate language if no text detected
- **Multiple Providers**: PaddleOCR, Tesseract, Google Gemini, OpenAI GPT-4 Vision
- **Smart Receipt Parsing**: Automatically extracts merchant name, total amount, date, and category
- **Line Item Detection**: Identifies individual items and their prices
- **REST API**: Simple JSON API for seamless integration

## 📋 Prerequisites

- Python 3.10+
- 2GB+ RAM (for PaddleOCR models)
- pip (Python package manager)

## 🛠️ Quick Start

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```env
# AI Provider: paddle (default), tesseract, gemini, or openai
AI_PROVIDER=paddle

# Default OCR language: en (English) or km (Khmer)
OCR_LANGUAGE=en

# Django settings
SECRET_KEY=your-secret-key
DEBUG=True
```

### 4. Run the Server

```bash
python manage.py runserver 0.0.0.0:5001
```

## 📚 API Endpoints

### Scan Receipt

```http
POST /api/ocr/scan
Content-Type: application/json

{
    "image": "base64_encoded_image_data",
    "language": "en"  // or "km" for Khmer
}
```

**Success Response:**
```json
{
    "success": true,
    "merchant": "Walmart",
    "total": 45.99,
    "date": "2025-11-30",
    "category": "Shopping",
    "items": [
        {"name": "Milk", "price": 3.99}
    ],
    "raw_text": "...",
    "language": "en",
    "provider": "paddle"
}
```

### Get Provider Info

```http
GET /api/ocr/provider
```

**Response:**
```json
{
    "current_provider": "paddle",
    "default_language": "en",
    "provider_info": {
        "name": "PaddleOCR",
        "type": "local",
        "description": "High-accuracy local OCR engine with English and Khmer support",
        "requires_api_key": false,
        "supported_languages": ["en", "km"]
    },
    "available_providers": ["paddle", "tesseract", "gemini", "openai"]
}
```

### Health Check

```http
GET /api/health
```

## 🌐 Language Support

| Language | Code | Model |
|----------|------|-------|
| English | `en` | PaddleOCR English |
| Khmer (ខ្មែរ) | `km` | PaddleOCR Khmer |

### Usage Example

**English Receipt:**
```bash
curl -X POST http://localhost:5001/api/ocr/scan \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_data_here", "language": "en"}'
```

**Khmer Receipt:**
```bash
curl -X POST http://localhost:5001/api/ocr/scan \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_data_here", "language": "km"}'
```

## 🤖 AI Provider Comparison

| Provider | Accuracy | Speed | Cost | Languages |
|----------|----------|-------|------|-----------|
| **PaddleOCR** | High | Fast | Free | en, km |
| **Tesseract** | Medium | Fast | Free | en |
| **Gemini** | Very High | Medium | Pay-per-use | Multilingual |
| **OpenAI** | Highest | Medium | Pay-per-use | Multilingual |

## 📊 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AI_PROVIDER` | Active OCR provider | `paddle` |
| `OCR_LANGUAGE` | Default language | `en` |
| `GOOGLE_API_KEY` | Google Gemini API key | - |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `SECRET_KEY` | Django secret key | Auto-generated |
| `DEBUG` | Debug mode | `False` |

---

## 🚀 Deployment Guide

See [DEPLOYMENT.md](../DEPLOYMENT.md) for complete deployment instructions for all services.

### Quick Deploy to DigitalOcean

```bash
# SSH to your droplet
ssh root@your-droplet-ip

# Run deployment script
curl -sSL https://raw.githubusercontent.com/YOUR_USER/fint-ai/main/deploy.sh | bash
```

### Pre-download OCR Models

For faster startup, pre-download both language models:

```bash
python << 'EOF'
from paddleocr import PaddleOCR
print("Downloading English model...")
PaddleOCR(use_angle_cls=True, lang='en', show_log=True)
print("Downloading Khmer model...")
PaddleOCR(use_angle_cls=True, lang='km', show_log=True)
print("Done!")
EOF
```

## 🐳 Docker Deployment

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download OCR models
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en'); PaddleOCR(lang='km')"

COPY . .
EXPOSE 5001
CMD ["gunicorn", "--config", "gunicorn.conf.py", "fint_ai.wsgi:application"]
```

## 📄 License

MIT License
