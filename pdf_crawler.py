#!/usr/bin/env python3
"""
Smart PDF Crawler & Compiler.

This script performs a Breadth-First Search (BFS) on a website, 
finds PDF documents, and uses Google Gemini to extract information 
related to a specific KEYWORD.

It produces a single "Knowledge Compilation" report instead of a folder of files.

Usage:
    python pdf_crawler.py "https://example.com/docs/" --keyword "revenue" --threads 10
"""

import os
import time
import argparse
import requests
import re
import threading
import signal
import sys
import json
from urllib.parse import urljoin, urlparse, unquote
from collections import deque
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict, Optional, Any

# Rich UI Imports
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich import box

# Intelligent Analysis Module
try:
    from gemini_analyzer import GeminiAnalyzer
except ImportError:
    print("Error: gemini_analyzer.py not found or dependencies missing.")
    sys.exit(1)

# ==============================================================================
# GLOBAL CONFIGURATION & STATE MANAGEMENT
# ==============================================================================

class CrawlerState:
    def __init__(self):
        self.lock = threading.Lock()
        
        # Stats
        self.scanned: int = 0
        self.analyzed_count: int = 0
        self.relevant_count: int = 0
        self.errors: int = 0
        self.total_content_bytes: int = 0
        
        # Findings Storage
        # List of {topic: str, url: str, quote: str, summary: str}
        self.findings: List[Dict[str, str]] = []
        
        # UI Logs
        self.activity_log: deque = deque(maxlen=8)
        self.last_finding: str = "None yet..."
        
        # Concurrency
        self.active_tasks: int = 0
        self.concurrency_limit: int = 5
        self.max_concurrency_cap: int = 20
        
        # Navigation
        self.visited: Set[str] = set()
        self.queue: deque = deque()
        self.start_url: str = ""
        self.keywords: List[str] = []
        
        self.is_running = True

    def save_state(self, filename: str):
        data = {
            "scanned": self.scanned,
            "analyzed_count": self.analyzed_count,
            "relevant_count": self.relevant_count,
            "total_content_bytes": self.total_content_bytes,
            "findings": self.findings,
            "visited": list(self.visited),
            "queue": list(self.queue),
            "start_url": self.start_url,
            "keywords": self.keywords
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"\n[+] State saved to {filename}")

    def load_state(self, filename: str) -> bool:
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            self.scanned = data.get("scanned", 0)
            self.analyzed_count = data.get("analyzed_count", 0)
            self.relevant_count = data.get("relevant_count", 0)
            self.total_content_bytes = data.get("total_content_bytes", 0)
            self.findings = data.get("findings", [])
            self.visited = set(data.get("visited", []))
            self.queue = deque(data.get("queue", []))
            self.start_url = data.get("start_url", "")
            self.keywords = data.get("keywords", [])
            print(f"[+] Resumed state from {filename}. Queue size: {len(self.queue)}")
            return True
        except FileNotFoundError:
            print(f"[-] State file {filename} not found. Starting fresh.")
            return False
        except Exception as e:
            print(f"[-] Failed to load state: {e}")
            return False

state = CrawlerState()

def log_activity(message: str):
    """Thread-safe UI logger."""
    with state.lock:
        state.activity_log.append(message)

# ==============================================================================
# WORKER LOGIC
# ==============================================================================

def process_url(session: requests.Session, analyzer: GeminiAnalyzer, url: str, scope_url: str) -> List[str]:
    """
    Worker function.
    """
    new_links = []
    
    try:
        with state.lock:
            state.active_tasks += 1
            
        response = session.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            log_activity(f"[cyan]Analyzing PDF:[/cyan] {url.split('/')[-1]}")
            pdf_bytes = response.content
            
            text = analyzer.extract_text(pdf_bytes)
            # Pass list of keywords
            results = analyzer.analyze_content(text, state.keywords)
            
            with state.lock:
                state.analyzed_count += 1
                if results:
                    state.relevant_count += len(results)
                    for res in results:
                        # Calculate estimated size (Quote + Summary + URL + Formatting overhead)
                        size_est = len(res.get('quote', '')) + len(res.get('summary', '')) + len(url) + 50
                        state.total_content_bytes += size_est
                        
                        state.findings.append({
                            "url": url,
                            "topic": res.get('topic', 'General'),
                            "quote": res.get('quote', ''),
                            "summary": res.get('summary', '')
                        })
                    
                    # Log the first finding as a sample
                    first = results[0]
                    state.last_finding = f"[green]Found ({first.get('topic')}):[/green] {first.get('summary')[:80]}..."
                    log_activity(f"[green bold]FOUND {len(results)} RELEVANT ITEMS![/green bold]")
                
        elif 'text/html' in content_type:
            log_activity(f"[dim]Scanning:[/dim] {url.split('/')[-1]}")
            try:
                html_content = response.content
                if len(html_content) > 5 * 1024 * 1024:
                    raise ValueError("HTML too large")
                    
                soup = BeautifulSoup(html_content, 'html.parser')
                base_domain = urlparse(scope_url).netloc
                
                for link in soup.find_all('a', href=True):
                    abs_url = urljoin(url, link['href']).split('#')[0]
                    
                    # Logic: Allow if strictly in scope OR if it is a PDF on the same domain
                    is_in_scope = abs_url.startswith(scope_url)
                    is_same_domain_pdf = abs_url.lower().endswith('.pdf') and urlparse(abs_url).netloc == base_domain
                    
                    if is_in_scope or is_same_domain_pdf:
                        new_links.append(abs_url)
                        
                with state.lock:
                    state.scanned += 1
            except Exception:
                pass

    except Exception as e:
        with state.lock:
            state.errors += 1
            state.concurrency_limit = max(1, state.concurrency_limit // 2)
            log_activity(f"[red]Error:[/red] {str(e)[:30]}...")
            
    finally:
        with state.lock:
            state.active_tasks -= 1
            
    return new_links

# ==============================================================================
# UI & REPORT COMPILER
# ==============================================================================

def generate_dashboard() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="stats", ratio=1),
        Layout(name="log", ratio=2)
    )

    # Header
    kws = ", ".join(state.keywords)
    layout["header"].update(
        Panel(f"Smart PDF Crawler - Keywords: [bold yellow]'{kws}'[/bold yellow]", 
              style="bold white on blue")
    )

    # Stats Table
    table = Table(box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    with state.lock:
        # Format bytes
        size_str = f"{state.total_content_bytes / 1024:.1f} KB"
        if state.total_content_bytes > 1024 * 1024:
            size_str = f"{state.total_content_bytes / (1024*1024):.1f} MB"
            
        table.add_row("Scanned Pages", str(state.scanned))
        table.add_row("PDFs Analyzed", str(state.analyzed_count))
        table.add_row("Relevant Findings", f"[bold green]{state.relevant_count}[/bold green]")
        table.add_row("Est. Report Size", f"[yellow]{size_str}[/yellow]")
        table.add_row("Errors", f"[red]{state.errors}[/red]")
        table.add_row("Active Threads", f"{state.active_tasks}/{state.concurrency_limit}")
        table.add_row("Queue Size", str(len(state.queue)))

    layout["stats"].update(Panel(table, title="Statistics", border_style="cyan"))

    # Activity Log
    log_text = Text()
    with state.lock:
        for msg in state.activity_log:
            log_text.append(Text.from_markup(msg + "\n"))
            
    layout["log"].update(Panel(log_text, title="Live Activity", border_style="green"))

    # Footer (Last Finding)
    with state.lock:
        footer_content = state.last_finding
    layout["footer"].update(Panel(footer_content, title="Latest Insight", border_style="yellow"))

    return layout

def compile_final_report(output_file: str = "Knowledge_Compilation.md"):
    console = Console()
    console.print(f"\n[bold blue][*] Compiling report to {output_file}...[/bold blue]")
    
    # Group by Topic
    from collections import defaultdict
    grouped = defaultdict(list)
    for f in state.findings:
        grouped[f['topic']].append(f)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Knowledge Compilation\n\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Keywords**: {', '.join(state.keywords)}\n")
        f.write(f"**Source**: {state.start_url}\n")
        f.write(f"**Total Findings**: {state.relevant_count}\n\n")
        f.write("---\n\n")
        
        if not state.findings:
            f.write("> No relevant information found.\n")
        else:
            for topic, items in grouped.items():
                f.write(f"## Topic: {topic}\n\n")
                for item in items:
                    f.write(f"### Source: [{item['url'].split('/')[-1]}]({item['url']})\n")
                    f.write(f"**Summary**: {item['summary']}\n\n")
                    f.write(f"> \"{item['quote']}\"\n\n")
                    f.write("---\n\n")
                
    console.print(f"[bold green][*] Report saved successfully.[/bold green]")

# ==============================================================================
# MAIN CONTROLLER
# ==============================================================================

def signal_handler(signum, frame):
    state.is_running = False

def crawl(start_url: str, keywords: List[str], max_threads: int, resume: bool = False):
    signal.signal(signal.SIGINT, signal_handler)
    
    # Init Analyzer
    try:
        analyzer = GeminiAnalyzer()
    except Exception as e:
        print(f"[!] Failed to init Analyzer: {e}")
        return

    # Init State
    loaded = False
    if resume:
        loaded = state.load_state("crawler_state.json")
    
    if not loaded:
        state.queue.append(start_url)
        state.visited.add(start_url)
        state.start_url = start_url
        state.keywords = keywords
        state.max_concurrency_cap = max_threads
    
    # Ensure some critical state is synced if we loaded (keywords could be overwritten if user provided different ones, 
    # but for resume strictness we usually keep the old ones or update? Let's keep old ones if loaded, or override?
    # The plan said "continue exactly where it left off, forcing the same URL and keywords". 
    # load_state sets self.keywords. So we should NOT override it unless we strictly want to allow changing keywords mid-flight.
    # But for now, let's respect the loaded state if it exists.
    if loaded:
        # Update concurrency cap from current args even if resumed
        state.max_concurrency_cap = max_threads
        # Recalculate keywords string for display
        keywords = state.keywords
        start_url = state.start_url
        
    scope_url = start_url if start_url.endswith('/') else start_url + '/'
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=max_threads, pool_maxsize=max_threads)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    futures = set()
    
    # Live Dashboard Loop
    kws_display = ", ".join(keywords)
    print(f"[*] Starting Multi-Keyword Crawl for: {kws_display}")
    time.sleep(1)
    
    executor = ThreadPoolExecutor(max_workers=max_threads)
    
    try:
        with Live(generate_dashboard(), refresh_per_second=4, screen=False) as live:
            
            while state.is_running and (state.queue or futures):
                
                # Update Dashboard
                live.update(generate_dashboard())
                
                # AICD: Additive Increase Concurrency
                with state.lock:
                    if state.active_tasks >= state.concurrency_limit - 1:
                        if state.concurrency_limit < state.max_concurrency_cap:
                            state.concurrency_limit += 1
                
                # Fill Queue
                with state.lock:
                    while len(state.queue) > 0 and state.active_tasks < state.concurrency_limit:
                        url = state.queue.popleft()
                        future = executor.submit(process_url, session, analyzer, url, scope_url)
                        futures.add(future)
                
                # Check Results
                completed = [f for f in futures if f.done()]
                futures.difference_update(completed)
                
                for f in completed:
                    try:
                        new_links = f.result()
                        with state.lock:
                            for link in new_links:
                                if link not in state.visited:
                                    state.visited.add(link)
                                    state.queue.append(link)
                    except Exception as e:
                        log_activity(f"[red]Fatal Worker Error:[/red] {e}")
                
                time.sleep(0.1)
                
    except KeyboardInterrupt:
        print("\n[!] Force stopping crawler (Ctrl+C detected)...")
        state.is_running = False
        state.save_state("crawler_state.json")
        compile_final_report()
        import os
        os._exit(0)
    finally:
        state.is_running = False
        executor.shutdown(wait=False)
            
    compile_final_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('url')
    # CHANGED: 'keywords' with nargs='+'
    parser.add_argument('--keywords', '-k', nargs='+', help="Topics to extract (required unless resuming)")
    parser.add_argument('--threads', '-t', type=int, default=5)
    parser.add_argument('--resume', action='store_true', help="Resume from crawler_state.json")
    args = parser.parse_args()
    
    if not args.resume and not args.keywords:
        parser.error("--keywords is required unless --resume is specified")
    
    # If resuming, keywords might be None, but crawl handles it
    kw_arg = args.keywords if args.keywords else []
    
    crawl(args.url, kw_arg, args.threads, args.resume)
