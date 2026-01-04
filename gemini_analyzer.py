
import os
import json
from pypdf import PdfReader
from io import BytesIO
from google import genai
from google.genai import types

class GeminiAnalyzer:
    def __init__(self):
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
            
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = 'gemini-3-flash-preview'

    def extract_text(self, pdf_bytes: bytes) -> str:
        """Extracts text from PDF bytes using pypdf."""
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""

    def analyze_content(self, text: str, keywords: list[str], logger=None) -> list[dict]:
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

        import time
        import random

        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries + 1):
            try:
                # Use streaming to provide immediate feedback
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                full_text = ""
                first_chunk_received = False
                
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        text_chunk = chunk.text
                        full_text += text_chunk
                        if not first_chunk_received:
                            msg = f"[cyan]Receiving AI response...[/cyan] (Token stream started)"
                            if logger: logger(msg)
                            elif attempt == 0: print(msg) # Only print if not retrying to avoid spam
                            first_chunk_received = True

                try:
                    import json
                    result = json.loads(full_text)
                except Exception:
                    # Fallback if response text isn't raw JSON string or incomplete
                    msg = f"Debug: Failed to parse JSON. Text len: {len(full_text)}"
                    if logger: logger(f"[dim]{msg}[/dim]")
                    else: print(msg)
                    return []
                
                # Ensure it's a list
                if isinstance(result, list):
                    return result
                # Handle edge case where model returns single object
                if isinstance(result, dict) and 'topic' in result:
                    return [result]
                    
                return []

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    if attempt < max_retries:
                        delay = (base_delay * (2 ** attempt)) + (random.random() * 1.0)
                        msg = f"[!] API Rate Limit (429). Retrying in {delay:.2f}s..."
                        if logger: logger(f"[yellow]{msg}[/yellow]") 
                        else: print(msg)
                        time.sleep(delay)
                        continue
                    else:
                        msg = "[!] Max retries reached for API Limit."
                        if logger: logger(f"[red]{msg}[/red]")
                        else: print(msg)
                        return []
                elif "404" in error_str or "Not Found" in error_str:
                    msg = f"[-] Model {self.model_name} not found (404). Skipping."
                    if logger: logger(f"[red]{msg}[/red]")
                    else: print(msg)
                    return []
                else:
                    msg = f"AI Analysis failed: {e}"
                    if logger: logger(f"[red]{msg}[/red]")
                    else: print(msg)
                    return []
        return []
