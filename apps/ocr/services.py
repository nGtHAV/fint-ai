"""
OCR Service - Multiple AI provider support with Surya OCR
"""
import base64
import io
import re
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from PIL import Image
from django.conf import settings
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)


class OCRProvider(ABC):
    """Abstract base class for OCR providers"""
    
    @abstractmethod
    def extract_receipt_data(self, image_data: bytes) -> dict:
        """Extract receipt data from image"""
        pass


class SuryaOCR_Provider(OCRProvider):
    """Surya OCR provider - Modern, efficient, GPU-optimized (~2-3GB VRAM)"""
    
    _instance = None
    _foundation = None
    _detection = None
    _recognition = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _init_predictors(self):
        """Lazy load Surya OCR predictors"""
        if self._foundation is None:
            print("[OCR] Initializing Surya OCR (optimized for 6GB VRAM)...", flush=True)
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.detection import DetectionPredictor
            
            # Initialize in correct order
            self._foundation = FoundationPredictor()
            self._detection = DetectionPredictor()
            self._recognition = RecognitionPredictor(self._foundation)
            print("[OCR] Surya OCR initialized successfully", flush=True)
    
    def extract_receipt_data(self, image_data: bytes, lang: str = None) -> dict:
        print(f"\n[OCR] Starting receipt extraction with Surya OCR...", flush=True)
        try:
            # Open image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Initialize predictors
            self._init_predictors()
            
            # Run recognition with detection predictor (Surya v0.17+ API)
            # The recognition predictor will handle detection internally
            print("[OCR] Running OCR (detection + recognition)...", flush=True)
            rec_predictions = self._recognition(
                [image], 
                det_predictor=self._detection
            )
            
            if rec_predictions and len(rec_predictions) > 0:
                lines = []
                pred = rec_predictions[0]
                
                print(f"[OCR] Surya found {len(pred.text_lines)} text lines", flush=True)
                
                for text_line in pred.text_lines:
                    text = text_line.text
                    confidence = text_line.confidence
                    print(f"[OCR] Text: '{text}' (confidence: {confidence:.2f})", flush=True)
                    if confidence > 0.3:
                        lines.append(text)
                
                full_text = '\n'.join(lines)
                result_data = self._parse_receipt_text(full_text, lines)
                result_data['language'] = lang or 'en'
                
                # Log the result
                print(f"\n{'='*50}", flush=True)
                print(f"[OCR] Parsed result:", flush=True)
                print(json.dumps(result_data, indent=2, default=str), flush=True)
                print(f"{'='*50}\n", flush=True)
                
                return result_data
            else:
                print(f"[OCR] No text detected in image", flush=True)
                return {
                    'success': False,
                    'error': 'No text detected in image',
                    'raw_text': ''
                }
                
        except Exception as e:
            import traceback
            print(f"\n[OCR] Surya OCR failed: {str(e)}", flush=True)
            print(traceback.format_exc(), flush=True)
            return {
                'success': False,
                'error': f'Surya OCR failed: {str(e)}',
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
            'Food & Dining': ['restaurant', 'cafe', 'coffee', 'food', 'pizza', 'burger', 'sushi', 'diner', 'kitchen', 'grill', 'bakery', 'starbucks', 'mcdonald', 'subway', 'chipotle', 'wendy', 'taco', 'kfc', 'popeyes', 'tea', 'milk'],
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


class EasyOCR_Provider(OCRProvider):
    """EasyOCR provider (local, free, high accuracy) - English and Khmer support"""
    
    _instance = None
    _reader = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_reader(self):
        """Lazy load EasyOCR reader"""
        if self._reader is None:
            import easyocr
            print("[OCR] Initializing EasyOCR with GPU...", flush=True)
            # Support English (Khmer 'km' is not supported by EasyOCR)
            # For Khmer text, consider using Tesseract with khm language pack
            self._reader = easyocr.Reader(['en'], gpu=True)
            print("[OCR] EasyOCR initialized successfully with GPU", flush=True)
        return self._reader
    
    def extract_receipt_data(self, image_data: bytes, lang: str = None) -> dict:
        print(f"\n[OCR] Starting receipt extraction with EasyOCR...", flush=True)
        try:
            # Open image and convert to numpy array
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            
            # Get OCR reader
            reader = self._get_reader()
            
            # Run OCR - EasyOCR returns: [[bbox, text, confidence], ...]
            result = reader.readtext(image_np)
            
            print(f"[OCR] EasyOCR found {len(result)} text regions", flush=True)
            
            if result:
                lines = []
                for detection in result:
                    # EasyOCR format: [bbox, text, confidence]
                    bbox, text, confidence = detection
                    print(f"[OCR] Text: '{text}' (confidence: {confidence:.2f})", flush=True)
                    if confidence > 0.3:  # Filter very low confidence
                        lines.append(text)
                
                full_text = '\n'.join(lines)
                result_data = self._parse_receipt_text(full_text, lines)
                result_data['language'] = lang or 'multi'
                
                # Log the result
                print(f"\n{'='*50}", flush=True)
                print(f"[OCR] Parsed result:", flush=True)
                print(json.dumps(result_data, indent=2, default=str), flush=True)
                print(f"{'='*50}\n", flush=True)
                
                return result_data
            else:
                print(f"[OCR] No text detected in image", flush=True)
                return {
                    'success': False,
                    'error': 'No text detected in image',
                    'raw_text': ''
                }
                
        except Exception as e:
            import traceback
            print(f"\n[OCR] EasyOCR failed: {str(e)}", flush=True)
            print(traceback.format_exc(), flush=True)
            return {
                'success': False,
                'error': f'EasyOCR failed: {str(e)}',
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
    provider = getattr(settings, 'AI_PROVIDER', 'surya').lower()
    
    if provider == 'gemini':
        return GeminiOCR()
    elif provider == 'openai':
        return OpenAIOCR()
    elif provider == 'tesseract':
        return TesseractOCR()
    elif provider == 'easyocr':
        return EasyOCR_Provider()
    else:
        # Default to Surya OCR (best for 6GB VRAM)
        return SuryaOCR_Provider()
