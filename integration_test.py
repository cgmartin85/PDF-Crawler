
import threading
import time
import os
import shutil
import mock_server
from unittest.mock import MagicMock
import sys

# Mock google.genai
# We have to mock it BEFORE importing gemini_analyzer
mock_google = MagicMock()
sys.modules['google'] = mock_google
sys.modules['google.genai'] = mock_google.genai

import gemini_analyzer

class MockAnalyzer:
    def extract_text(self, pdf_bytes): return "Text with Revenue"
    def analyze_content(self, text, keywords):
        return [{"topic": "Revenue", "quote": "Info", "summary": "Found info"}]

import pdf_crawler
pdf_crawler.GeminiAnalyzer = MockAnalyzer

def run_test():
    print("Starting Mock Server...")
    t = threading.Thread(target=mock_server.start_server)
    t.daemon = True
    t.start()
    time.sleep(1)
    
    print("Starting Crawler...")
    try:
        # Port 8890 from last time
        pdf_crawler.crawl("http://localhost:8890", ["Revenue"], 2)
    except Exception:
        pass

    if os.path.exists("mock_site"):
        shutil.rmtree("mock_site")

if __name__ == "__main__":
    run_test()
