
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

    def extract_images(self, pdf_bytes: bytes, max_images: int = 5) -> list[dict]:
        """
        Extracts images from PDF bytes.
        Returns list of dicts: {'mime_type': 'image/jpeg', 'data': bytes}
        """
        images = []
        try:
            reader = PdfReader(BytesIO(pdf_bytes))
            for page in reader.pages:
                if len(images) >= max_images:
                    break
                
                # Check for images in page resources
                if '/XObject' in page.get('/Resources', {}):
                    xObject = page['/Resources']['/XObject'].get_object()
                    for obj in xObject:
                        if xObject[obj]['/Subtype'] == '/Image':
                            img_obj = xObject[obj]
                            # Filter small icons/logos if possible (by size?) 
                            # For now just grab them.
                            # pypdf creates image file automatically? No, we need data.
                            
                            # Simplest: use page.images (pypdf >= 3.0.0)
                            pass
            
            # Better approach with modern pypdf:
            for page in reader.pages:
                if len(images) >= max_images:
                    break
                for image_file_object in page.images:
                    if len(images) >= max_images: 
                        break
                    
                    # image_file_object.data is bytes
                    # image_file_object.name helps guess extension/mime
                    
                    mime = "image/jpeg"
                    if image_file_object.name.lower().endswith('.png'):
                        mime = "image/png"
                    
                    # Skip very small images (icons) - e.g. < 5KB
                    if len(image_file_object.data) < 5 * 1024:
                        continue
                        
                    images.append({
                        'mime_type': mime,
                        'data': image_file_object.data
                    })
                    
            return images
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []

    def analyze_images(self, images: list[dict], keywords: list[str], logger=None, usage_callback=None) -> list[dict]:
        """
        Sends images to Gemini to identify presence of keywords/people.
        """
        if not images:
            return []
            
        # Construct content parts for Gemini
        # Prompt + Images
        
        prompt_text = f"""
        You are an image analyst.
        Attached are images extracted from a document.
        
        Your goal is to identify if ANY of the following people or topics are visually present in these images: {keywords}
        
        For EACH positive identification:
        1. Identify the TOPIC/PERSON.
        2. Describe the IMAGE content briefly (1 sentence).
        3. Quote "Visual Match".
        
        Format your response exactly as a JSON LIST of objects.
        Example:
        [
            {{
                "topic": "Donald Trump",
                "quote": "Visual Match",
                "summary": "Photo shows Donald Trump speaking at a podium."
            }}
        ]
        
        If none are found, return [].
        """
        
        contents = [prompt_text]
        for img in images:
            contents.append(types.Part.from_bytes(data=img['data'], mime_type=img['mime_type']))
            
        import time
        import random
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries + 1):
            try:
                # Use generate_content (non-stream for images is usually safer/simpler?) 
                # Or stream? Let's use stream to stay consistent.
                
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                full_text = ""
                first_chunk_received = False
                
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        full_text += chunk.text
                        if not first_chunk_received:
                            if logger: logger("[cyan]Analyzing visual content...[/cyan]")
                            first_chunk_received = True

                    # Capture Usage Metadata for Vision
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        if usage_callback:
                            u = chunk.usage_metadata
                            usage_callback(u.prompt_token_count, u.candidates_token_count)
                            
                import json
                result = json.loads(full_text)
                if isinstance(result, list):
                    return result
                if isinstance(result, dict) and 'topic' in result:
                    return [result]
                return []
                
            except Exception as e:
                # Simple retry logic similar to analyze_content
                if attempt < max_retries:
                    time.sleep(2)
                    continue
                if logger: logger(f"[red]Image Analysis failed:[/red] {e}")
                return []

        return []

    def analyze_content(self, text: str, keywords: list[str], logger=None, usage_callback=None) -> list[dict]:
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

                    # Capture Usage Metadata
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        if usage_callback:
                            u = chunk.usage_metadata
                            usage_callback(u.prompt_token_count, u.candidates_token_count)
                            
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

    def generate_executive_summary(self, findings: list[dict], source_name: str, logger=None, usage_callback=None) -> str:
        """
        Generates a high-level executive summary for a specific directory/source.
        """
        if not findings:
            return "No findings to summarize."

        # Prepare context
        context_text = f"Source: {source_name}\n\nFindings:\n"
        for i, f in enumerate(findings[:50]): # Limit to first 50 findings to avoid massive context key overflow? 
            context_text += f"- [{f['topic']}] {f['summary']} (Ref: {f['url']})\n"
            
        prompt = f"""
        You are an Intelligence Analyst. 
        Read the following extracted findings from "{source_name}".
        
        Write a concise Executive Summary (2-3 paragraphs) that:
        1. Identifies the main themes/topics found in this directory.
        2. Highlights any critical or specific individuals mentioned (especially regarding the keywords).
        3. Provides a high-level interpretation of what these documents represent.
        
        Do not list every finding. Synthesize the information.
        
        {context_text}
        """
        
        try:
             # Just use generate_content for simplicity (not stream)
             response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
             )
             
             if response.usage_metadata and usage_callback:
                 usage_callback(response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
                 
             return response.text
             
        except Exception as e:
            msg = f"Summary Generation Failed: {e}"
            if logger: logger(f"[red]{msg}[/red]")
            return f"> **Error generating summary**: {e}"
