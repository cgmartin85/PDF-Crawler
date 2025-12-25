import http.server
import socketserver
import threading
import os
import time

PORT = 8890

class MockHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

def start_server():
    os.makedirs("mock_site/files", exist_ok=True)
    
    with open("mock_site/index.html", "w") as f:
        f.write('<html><body><a href="doc1.pdf">Doc 1</a></body></html>')

    try:
        from reportlab.pdfgen import canvas
        def create_pdf(path, content):
            c = canvas.Canvas(path)
            c.drawString(100, 750, content)
            c.save()
        create_pdf("mock_site/doc1.pdf", "Inside Scope Revenue")
    except ImportError:
        with open("mock_site/doc1.pdf", "wb") as f: f.write(b"%PDF-1.4 dummy")

    os.chdir("mock_site")
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), MockHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    start_server()
