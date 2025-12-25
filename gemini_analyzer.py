import os
import io
import google.generativeai as genai
from pypdf import PdfReader
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

class GeminiAnalyzer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extracts text from PDF bytes using pypdf."""
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def analyze_content(self, text: str, keywords: list[str]) -> list[dict]:
        """
        Sends text to Gemini to extract sections related to ANY of the keywords.
        Returns a list of dicts: [{'topic': '...', 'quote': '...', 'summary': '...'}]
        """
        if not text.strip():
            return []

        # Pre-filter: Check if ANY keyword is in text (case-insensitive)
        text_lower = text.lower()
        if not any(k.lower() in text_lower for k in keywords):
            return []

        prompt = f"""
        You are a research assistant. 
        Read the following text from a document.
        
        Your goal is to identify sections relevant to ANY of these topics: {keywords}
        
        For EACH relevant section you find:
        1. Identify the specific TOPIC it matches (from the list above).
        2. Extract a direct QUOTE.
        3. Write a 1-sentence SUMMARY.
        
        Format your response exactly as a JSON LIST of objects. 
        Example:
        [
            {{
                "topic": "Revenue",
                "quote": "Revenue grew by 5%...",
                "summary": "Q3 revenue saw positive growth."
            }}
        ]
        
        If NO relevant information is found for any topic, return an empty list: []
        
        Text to analyze:
        {text[:25000]} 
        """ 

        try:
            response = self.model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            import json
            result = json.loads(response.text)
            
            # Ensure it's a list
            if isinstance(result, list):
                return result
            # Handle edge case where model returns single object
            if isinstance(result, dict) and 'topic' in result:
                return [result]
                
            return []
        except Exception as e:
            print(f"AI Analysis failed: {e}")
            return []
