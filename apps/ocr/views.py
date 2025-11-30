"""
OCR API views
"""
import base64
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

from .services import get_ocr_provider, SuryaOCR_Provider, EasyOCR_Provider


@api_view(['POST'])
def scan_receipt(request):
    """
    Scan a receipt image and extract data
    
    Request body:
    {
        "image": "base64 encoded image data",
        "language": "en" (optional, default: en)
    }
    
    Response:
    {
        "success": true,
        "merchant": "Store name",
        "total": 123.45,
        "date": "2025-11-30",
        "category": "Food & Dining",
        "items": [{"name": "Item", "price": 10.00}],
        "raw_text": "...",
        "language": "en"
    }
    """
    image_data = request.data.get('image')
    language = request.data.get('language', getattr(settings, 'OCR_LANGUAGE', 'en'))
    
    if not image_data:
        return Response(
            {'error': 'Image data is required'},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Check file size
        max_size = getattr(settings, 'MAX_UPLOAD_SIZE', 16 * 1024 * 1024)
        if len(image_bytes) > max_size:
            return Response(
                {'error': 'Image size exceeds maximum allowed (16MB)'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get OCR provider and extract data
        provider = get_ocr_provider()
        
        # Pass language to providers that support it
        if isinstance(provider, (SuryaOCR_Provider, EasyOCR_Provider)):
            result = provider.extract_receipt_data(image_bytes, lang=language)
        else:
            result = provider.extract_receipt_data(image_bytes)
        
        # Add provider name to response
        result['provider'] = getattr(settings, 'AI_PROVIDER', 'surya').lower()
        
        return Response(result)
        
    except Exception as e:
        return Response(
            {'error': f'Failed to process image: {str(e)}', 'success': False},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_provider_info(request):
    """Get information about the current OCR provider"""
    provider = getattr(settings, 'AI_PROVIDER', 'surya').lower()
    default_lang = getattr(settings, 'OCR_LANGUAGE', 'en')
    
    providers_info = {
        'surya': {
            'name': 'Surya OCR',
            'type': 'local',
            'description': 'Modern, efficient OCR optimized for documents. Uses ~2-3GB VRAM.',
            'requires_api_key': False,
            'supported_languages': ['en', 'multilingual'],
            'gpu_enabled': True,
            'vram_usage': '2-3GB'
        },
        'easyocr': {
            'name': 'EasyOCR',
            'type': 'local',
            'description': 'High-accuracy local OCR engine with GPU support',
            'requires_api_key': False,
            'supported_languages': ['en'],
            'gpu_enabled': True,
            'vram_usage': '3-4GB'
        },
        'tesseract': {
            'name': 'Tesseract OCR',
            'type': 'local',
            'description': 'Local OCR engine, free but less accurate',
            'requires_api_key': False,
            'supported_languages': ['en']
        },
        'gemini': {
            'name': 'Google Gemini',
            'type': 'cloud',
            'description': 'Google AI with vision capabilities, high accuracy',
            'requires_api_key': True,
            'supported_languages': ['multilingual']
        },
        'openai': {
            'name': 'OpenAI GPT-4 Vision',
            'type': 'cloud',
            'description': 'OpenAI with vision capabilities, highest accuracy',
            'requires_api_key': True,
            'supported_languages': ['multilingual']
        }
    }
    
    return Response({
        'current_provider': provider,
        'default_language': default_lang,
        'provider_info': providers_info.get(provider, providers_info['surya']),
        'available_providers': list(providers_info.keys())
    })
