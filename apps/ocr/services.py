"""
OCR Service - Multiple AI provider support with PaddleOCR
"""
import base64
import io
import re
import json
from abc import ABC, abstractmethod
from datetime import datetime
from PIL import Image
from django.conf import settings
import numpy as np


class OCRProvider(ABC):
    """Abstract base class for OCR providers"""
    
    @abstractmethod
    def extract_receipt_data(self, image_data: bytes) -> dict:
        """Extract receipt data from image"""
        pass


class PaddleOCR_Provider(OCRProvider):
    """PaddleOCR provider (local, free, high accuracy) - English and Khmer support"""
    
    _instance = None
    _ocr_en = None
    _ocr_km = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_ocr(self, lang='en'):
        """Lazy load PaddleOCR to avoid slow startup"""
        if lang == 'km' or lang == 'khmer':
            if self._ocr_km is None:
                from paddleocr import PaddleOCR
                # Khmer language OCR
                self._ocr_km = PaddleOCR(
                    use_angle_cls=True,
                    lang='km',  # Khmer
                    show_log=False,
                    use_gpu=False
                )
            return self._ocr_km
        else:
            if self._ocr_en is None:
                from paddleocr import PaddleOCR
                # English language OCR (default)
                self._ocr_en = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=False
                )
            return self._ocr_en
    
    def extract_receipt_data(self, image_data: bytes, lang: str = None) -> dict:
        try:
            # Open image and convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            # Determine language from settings or parameter
            ocr_lang = lang or getattr(settings, 'OCR_LANGUAGE', 'en')
            
            # Get OCR instance for the language
            ocr = self._get_ocr(ocr_lang)
            
            # Run OCR
            result = ocr.ocr(image_np, cls=True)
            
            # Extract text from result
            if result and result[0]:
                lines = []
                for line in result[0]:
                    if line[1]:  # Check if text exists
                        text = line[1][0]  # Get the text
                        confidence = line[1][1]  # Get confidence
                        if confidence > 0.5:  # Filter low confidence
                            lines.append(text)
                
                full_text = '\n'.join(lines)
                result_data = self._parse_receipt_text(full_text, lines)
                result_data['language'] = ocr_lang
                return result_data
            else:
                # Try with the other language if no text detected
                alt_lang = 'km' if ocr_lang == 'en' else 'en'
                ocr_alt = self._get_ocr(alt_lang)
                result = ocr_alt.ocr(image_np, cls=True)
                
                if result and result[0]:
                    lines = []
                    for line in result[0]:
                        if line[1]:
                            text = line[1][0]
                            confidence = line[1][1]
                            if confidence > 0.5:
                                lines.append(text)
                    
                    full_text = '\n'.join(lines)
                    result_data = self._parse_receipt_text(full_text, lines)
                    result_data['language'] = alt_lang
                    return result_data
                
                return {
                    'success': False,
                    'error': 'No text detected in image',
                    'raw_text': ''
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'PaddleOCR failed: {str(e)}',
                'raw_text': ''
            }
    
    def _parse_receipt_text(self, text: str, lines: list) -> dict:
        """Parse receipt text to extract structured data"""
        
        # Try to extract merchant name (usually first non-empty line)
        merchant = ''
        for line in lines[:3]:  # Check first 3 lines
            line = line.strip()
            if line and len(line) > 2 and not re.match(r'^[\d\s\-\/\.]+$', line):
                merchant = line
                break
        
        # Try to find total amount
        total = None
        total_patterns = [
            r'total[:\s]*\$?([\d,]+\.?\d*)',
            r'amount\s*due[:\s]*\$?([\d,]+\.?\d*)',
            r'grand\s*total[:\s]*\$?([\d,]+\.?\d*)',
            r'balance\s*due[:\s]*\$?([\d,]+\.?\d*)',
            r'subtotal[:\s]*\$?([\d,]+\.?\d*)',
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amounts = []
                for m in matches:
                    try:
                        amounts.append(float(m.replace(',', '')))
                    except ValueError:
                        pass
                if amounts:
                    total = max(amounts)
                    break
        
        # If no total found, look for largest dollar amount
        if total is None:
            all_amounts = re.findall(r'\$\s*([\d,]+\.?\d*)', text)
            if all_amounts:
                amounts = []
                for m in all_amounts:
                    try:
                        amounts.append(float(m.replace(',', '')))
                    except ValueError:
                        pass
                if amounts:
                    total = max(amounts)
        
        # Try to find date
        date = None
        date_patterns = [
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', lambda m: f"{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{2})', lambda m: f"20{m.group(3)}-{m.group(1).zfill(2)}-{m.group(2).zfill(2)}"),
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', lambda m: f"{m.group(1)}-{m.group(2).zfill(2)}-{m.group(3).zfill(2)}"),
        ]
        
        for pattern, formatter in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date = formatter(match)
                    break
                except:
                    pass
        
        # Extract line items (look for price patterns)
        items = []
        item_pattern = r'(.+?)\s+\$?([\d,]+\.\d{2})\s*$'
        for line in lines:
            match = re.match(item_pattern, line.strip())
            if match:
                item_name = match.group(1).strip()
                item_price = float(match.group(2).replace(',', ''))
                if item_name and item_price > 0 and item_price != total:
                    items.append({'name': item_name, 'price': item_price})
        
        # Guess category based on keywords
        category = self._guess_category(text)
        
        return {
            'success': True,
            'merchant': merchant,
            'total': total,
            'date': date,
            'category': category,
            'raw_text': text,
            'items': items[:10]  # Limit to 10 items
        }
    
    def _guess_category(self, text: str) -> str:
        """Guess category based on text content"""
        text_lower = text.lower()
        
        category_keywords = {
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'food', 'pizza', 'burger', 'sushi', 'diner', 'kitchen', 'grill', 'bakery', 'starbucks', 'mcdonald', 'subway', 'chipotle', 'wendy', 'taco', 'kfc', 'popeyes'],
            'Shopping': ['store', 'shop', 'mall', 'retail', 'amazon', 'walmart', 'target', 'costco', 'best buy', 'home depot', 'lowes', 'ikea', 'macy'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'parking', 'transit', 'metro', 'bus', 'shell', 'chevron', 'exxon', 'mobil', 'bp'],
            'Healthcare': ['pharmacy', 'hospital', 'clinic', 'medical', 'doctor', 'health', 'cvs', 'walgreens', 'rite aid'],
            'Entertainment': ['cinema', 'movie', 'theater', 'concert', 'ticket', 'game', 'netflix', 'spotify', 'amc', 'regal'],
            'Bills & Utilities': ['electric', 'water', 'internet', 'phone', 'utility', 'bill', 'verizon', 'at&t', 'comcast'],
            'Education': ['book', 'school', 'university', 'college', 'course', 'education', 'tuition', 'barnes'],
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return 'Other'


class TesseractOCR(OCRProvider):
    """Tesseract OCR provider (local, free) - Fallback"""
    
    def extract_receipt_data(self, image_data: bytes) -> dict:
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = getattr(settings, 'TESSERACT_CMD', '/usr/bin/tesseract')
            
            # Open image
            image = Image.open(io.BytesIO(image_data))
            
            # Extract text
            text = pytesseract.image_to_string(image)
            
            # Parse the extracted text
            return self._parse_receipt_text(text)
        except Exception as e:
            return {
                'success': False,
                'error': f'Tesseract OCR failed: {str(e)}',
                'raw_text': ''
            }
    
    def _parse_receipt_text(self, text: str) -> dict:
        """Parse receipt text to extract structured data"""
        lines = text.strip().split('\n')
        
        # Try to extract merchant name (usually first non-empty line)
        merchant = ''
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                merchant = line
                break
        
        # Try to find total amount
        total = None
        total_patterns = [
            r'total[:\s]*\$?([\d,]+\.?\d*)',
            r'amount[:\s]*\$?([\d,]+\.?\d*)',
            r'grand\s*total[:\s]*\$?([\d,]+\.?\d*)',
            r'\$\s*([\d,]+\.?\d*)',
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                amounts = []
                for m in matches:
                    try:
                        amounts.append(float(m.replace(',', '')))
                    except ValueError:
                        pass
                if amounts:
                    total = max(amounts)
                    break
        
        # Try to find date
        date = None
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',
            r'(\w+\s+\d{1,2},?\s+\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date = match.group(1)
                break
        
        # Guess category based on keywords
        category = self._guess_category(text)
        
        return {
            'success': True,
            'merchant': merchant,
            'total': total,
            'date': date,
            'category': category,
            'raw_text': text,
            'items': []
        }
    
    def _guess_category(self, text: str) -> str:
        """Guess category based on text content"""
        text_lower = text.lower()
        
        category_keywords = {
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'food', 'pizza', 'burger', 'sushi', 'diner', 'kitchen', 'grill', 'bakery'],
            'Transportation': ['gas', 'fuel', 'uber', 'lyft', 'taxi', 'parking', 'transit', 'metro', 'bus'],
            'Shopping': ['store', 'shop', 'mall', 'retail', 'amazon', 'walmart', 'target', 'costco'],
            'Healthcare': ['pharmacy', 'hospital', 'clinic', 'medical', 'doctor', 'health', 'cvs', 'walgreens'],
            'Entertainment': ['cinema', 'movie', 'theater', 'concert', 'ticket', 'game', 'netflix', 'spotify'],
            'Bills & Utilities': ['electric', 'water', 'gas', 'internet', 'phone', 'utility', 'bill'],
            'Education': ['book', 'school', 'university', 'college', 'course', 'education', 'tuition'],
        }
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category
        
        return 'Other'


class GeminiOCR(OCRProvider):
    """Google Gemini AI OCR provider"""
    
    def extract_receipt_data(self, image_data: bytes) -> dict:
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            prompt = """Analyze this receipt image and extract the following information in JSON format:
            {
                "merchant": "Store/restaurant name",
                "total": 0.00,
                "date": "YYYY-MM-DD",
                "category": "One of: Food & Dining, Transportation, Entertainment, Shopping, Healthcare, Bills & Utilities, Education, Other",
                "items": [
                    {"name": "Item name", "price": 0.00}
                ]
            }
            
            Only return the JSON object, no other text. If you cannot determine a value, use null."""
            
            response = model.generate_content([
                prompt,
                {
                    'mime_type': 'image/jpeg',
                    'data': image_b64
                }
            ])
            
            # Parse the response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = re.sub(r'^```\w*\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            data = json.loads(response_text)
            data['success'] = True
            data['raw_text'] = response_text
            
            return data
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Gemini OCR failed: {str(e)}',
                'raw_text': ''
            }


class OpenAIOCR(OCRProvider):
    """OpenAI Vision OCR provider"""
    
    def extract_receipt_data(self, image_data: bytes) -> dict:
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """Analyze this receipt image and extract the following information in JSON format:
                                {
                                    "merchant": "Store/restaurant name",
                                    "total": 0.00,
                                    "date": "YYYY-MM-DD",
                                    "category": "One of: Food & Dining, Transportation, Entertainment, Shopping, Healthcare, Bills & Utilities, Education, Other",
                                    "items": [
                                        {"name": "Item name", "price": 0.00}
                                    ]
                                }
                                Only return the JSON object, no other text. If you cannot determine a value, use null."""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = re.sub(r'^```\w*\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            data = json.loads(response_text)
            data['success'] = True
            data['raw_text'] = response_text
            
            return data
            
        except Exception as e:
            return {
                'success': False,
                'error': f'OpenAI OCR failed: {str(e)}',
                'raw_text': ''
            }


def get_ocr_provider() -> OCRProvider:
    """Get the configured OCR provider"""
    provider = getattr(settings, 'AI_PROVIDER', 'paddle').lower()
    
    if provider == 'gemini':
        return GeminiOCR()
    elif provider == 'openai':
        return OpenAIOCR()
    elif provider == 'tesseract':
        return TesseractOCR()
    else:
        # Default to PaddleOCR
        return PaddleOCR_Provider()
