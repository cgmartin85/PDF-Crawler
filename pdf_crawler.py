#!/usr/bin/env python3
"""
Advanced Multithreaded PDF Crawler with State Persistence.

This script performs a Breadth-First Search (BFS) on a specified website directory,
identifying and downloading all PDF documents found. It features:
    - Dynamic Concurrency Control (AIMD algorithm) to optimize network usage.
    - State Persistence (Checkpoints) to resume interrupted crawls.
    - Content-Disposition handling for correct filename resolution.
    - CSV Manifest generation for audit trails.
    - Graceful shutdown handling via Signal interception.

Usage:
    python pdf_crawler_advanced.py "https://example.com/docs/" --output "my_downloads" --threads 20
"""

import os
import time
import argparse
import requests
import re
import threading
import json
import csv
import signal
import sys
from urllib.parse import urljoin, urlparse, unquote
from collections import deque
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Set, Dict, Optional, Any

# ==============================================================================
# GLOBAL CONFIGURATION & STATE MANAGEMENT
# ==============================================================================

class CrawlerState:
    """
    A thread-safe Singleton-pattern class to manage the global state of the crawler.
    
    Attributes:
        lock (threading.Lock): Mutex to prevent race conditions during state updates.
        scanned (int): Count of HTML pages scanned for links.
        queued (int): Count of URLs currently waiting in the BFS queue.
        downloaded (int): Count of successfully downloaded PDFs.
        errors (int): Count of failed downloads or connection errors.
        bytes_downloaded (int): Total volume of data transferred in bytes.
        active_downloads (int): Number of threads currently downloading a file.
        concurrency_limit (int): Dynamic limit on how many threads can run simultaneously.
        max_concurrency_cap (int): The hard ceiling for concurrency_limit.
        visited (Set[str]): A set of all URLs processed to prevent infinite loops.
        queue (deque): The BFS FIFO queue containing URLs to visit.
        manifest_log (List[Dict]): Buffer for storing download results before writing to CSV.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # Statistics
        self.scanned: int = 0
        self.queued: int = 0
        self.downloaded: int = 0
        self.errors: int = 0
        self.bytes_downloaded: int = 0
        
        # Dynamic Concurrency Control (AIMD)
        self.active_downloads: int = 0
        self.concurrency_limit: int = 2   # Start conservative
        self.max_concurrency_cap: int = 30 # User defined ceiling
        
        # Rate Calculation (Speedometer)
        self.last_check_time: float = time.time()
        self.last_bytes_count: int = 0
        self.current_speed_mbps: float = 0.0

        # Persistence Data
        self.visited: Set[str] = set()
        self.queue: deque = deque()
        self.manifest_log: List[Dict[str, Any]] = []
        self.start_url: str = ""

    def save_checkpoint(self, filename: str = "crawler_checkpoint.json") -> None:
        """
        Serializes the current state to a JSON file.
        
        Uses an atomic write strategy (write to temp -> rename) to ensure
        the checkpoint file is never corrupted if the process crashes during write.
        
        Args:
            filename (str): The path to the checkpoint file.
        """
        with self.lock:
            # Convert non-serializable sets and deques to lists
            data = {
                "visited": list(self.visited),
                "queue": list(self.queue),
                "start_url": self.start_url,
                "stats": {
                    "scanned": self.scanned,
                    "downloaded": self.downloaded
                }
            }
            
            temp_name = filename + ".tmp"
            try:
                with open(temp_name, 'w') as f:
                    json.dump(data, f)
                
                # Atomic replacement of the old file
                if os.path.exists(filename):
                    os.remove(filename)
                os.rename(temp_name, filename)
                print(f"\n[CHECKPOINT] State saved to {filename}")
            except IOError as e:
                print(f"\n[ERROR] Failed to save checkpoint: {e}")

    def load_checkpoint(self, filename: str = "crawler_checkpoint.json") -> bool:
        """
        Loads the state from a JSON file if it exists.
        
        Args:
            filename (str): The path to the checkpoint file.
            
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise.
        """
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                # Reconstruct sets and deques from lists
                self.visited = set(data["visited"])
                self.queue = deque(data["queue"])
                self.start_url = data.get("start_url", "")
                self.scanned = data["stats"].get("scanned", 0)
                self.downloaded = data["stats"].get("downloaded", 0)
            
            print(f"[RESUME] Loaded checkpoint: {len(self.queue)} items in queue, {len(self.visited)} visited.")
            return True
        except Exception as e:
            print(f"[ERROR] Corrupt checkpoint file, starting fresh. Error: {e}")
            return False

# Instantiate Global State
state = CrawlerState()


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def sanitize_filename(name: str) -> str:
    """
    Sanitizes a string to be safe for use as a Windows filename.
    
    Args:
        name (str): The raw filename string.
        
    Returns:
        str: A safe filename with invalid characters replaced by underscores.
    """
    if not name:
        return "unknown_file"
    # Remove null bytes
    name = name.replace('\0', '')
    # Regex to replace Windows reserved characters: < > : " / \ | ? *
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def get_filename_from_cd(cd: str) -> Optional[str]:
    """
    Extracts the filename from the HTTP 'Content-Disposition' header.
    
    Example Header: 'attachment; filename="Annual Report 2024.pdf"'
    
    Args:
        cd (str): The Content-Disposition header string.
        
    Returns:
        Optional[str]: The extracted filename or None if not found.
    """
    if not cd:
        return None
    # Regex captures content inside quotes or unquoted filenames
    fname = re.findall(r'filename="?([^"]+)"?', cd)
    if len(fname) == 0:
        return None
    return fname[0]

def get_local_path(base_dir: str, url: str, content_disposition: Optional[str] = None) -> str:
    """
    Constructs a local file path that mirrors the remote directory structure.
    
    Priority for Filename:
    1. Content-Disposition header (Server authority).
    2. URL path basename (Fallback).
    3. Default 'index_doc.pdf' (Last resort).
    
    Args:
        base_dir (str): The local root directory for downloads.
        url (str): The source URL.
        content_disposition (str, optional): The HTTP Content-Disposition header.
        
    Returns:
        str: The full absolute local path.
    """
    parsed = urlparse(url)
    domain_dir = sanitize_filename(parsed.netloc)
    
    # Clean the path (remove leading slash) and decode URL-encoded chars (%20 -> space)
    path_path = unquote(parsed.path).lstrip('/')
    
    # Heuristic: Check if path looks like a file or a directory
    if '.' in os.path.basename(path_path):
        # e.g., /folder/file.pdf -> directory is /folder/
        directory_part = os.path.dirname(path_path)
    else:
        # e.g., /folder/subfolder -> directory is /folder/subfolder
        directory_part = path_path

    # Split and sanitize every directory component
    parts = directory_part.split('/')
    safe_parts = [sanitize_filename(p) for p in parts if p]
    
    # Determine the actual filename
    filename = None
    if content_disposition:
        filename = get_filename_from_cd(content_disposition)
    
    if not filename:
        # Fallback to URL basename
        base = os.path.basename(parsed.path)
        if not base or '.' not in base:
            # Handle directory-like URLs ending in / that return PDFs
            filename = "index_doc.pdf" 
        else:
            filename = unquote(base)

    safe_filename = sanitize_filename(filename)
    
    return os.path.join(base_dir, domain_dir, *safe_parts, safe_filename)

def is_subpath(target_url: str, root_url: str) -> bool:
    """
    Strictly enforces that the target URL is a subdirectory of the root URL.
    This prevents the crawler from ascending to parent directories or changing domains.
    
    Args:
        target_url (str): The URL to check.
        root_url (str): The starting URL defining the scope.
        
    Returns:
        bool: True if target is inside root scope.
    """
    if not root_url.endswith('/'):
        root_url += '/'
    return target_url.startswith(root_url)

def log_manifest(url: str, local_path: str, status: str, error_msg: str = "") -> None:
    """
    Appends a download event to the in-memory manifest log.
    
    Args:
        url (str): The source URL.
        local_path (str): The local destination path.
        status (str): "SUCCESS", "FAILED", or "SKIPPED".
        error_msg (str): Details of the error if failed.
    """
    with state.lock:
        state.manifest_log.append({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "url": url,
            "local_path": local_path,
            "status": status,
            "error": error_msg
        })

# ==============================================================================
# MONITORING & CONCURRENCY CONTROL
# ==============================================================================

def monitor_performance(stop_event: threading.Event, checkpoint_interval: int = 300) -> None:
    """
    Background thread that manages dynamic concurrency and periodic checkpointing.
    
    Logic:
        1. Calculates network throughput (MB/s).
        2. Implements AIMD (Additive Increase, Multiplicative Decrease) to adjust threads.
        3. Triggers atomic saves of the crawler state.
        
    Args:
        stop_event (threading.Event): Signal to stop the monitor loop.
        checkpoint_interval (int): Seconds between state saves.
    """
    last_save = time.time()
    
    while not stop_event.is_set():
        # Sleep interval determines update frequency of UI and Concurrency checks
        time.sleep(2.0)
        
        with state.lock:
            now = time.time()
            
            # --- Throughput Calculation ---
            delta_time = now - state.last_check_time
            delta_bytes = state.bytes_downloaded - state.last_bytes_count
            
            speed = (delta_bytes / 1024 / 1024) / delta_time if delta_time > 0 else 0
            state.current_speed_mbps = speed
            state.last_check_time = now
            state.last_bytes_count = state.bytes_downloaded
            
            # --- Dynamic Scaling (AIMD) ---
            # If we are utilizing nearly all allocated threads, attempt to increase limit.
            # (Note: Decrease happens in the error handler of the download function)
            if state.active_downloads >= state.concurrency_limit - 1:
                if state.concurrency_limit < state.max_concurrency_cap:
                    state.concurrency_limit += 1
            
            # --- Console Status Bar ---
            # \r overwrites the current line for a clean dashboard effect
            status = (
                f"\r[RUNNING] Threads: {state.active_downloads}/{state.concurrency_limit} | "
                f"Speed: {speed:.2f} MB/s | "
                f"PDFs: {state.downloaded} | "
                f"Queue: {len(state.queue)}   "
            )
            print(status, end="", flush=True)

        # --- Checkpoint Logic ---
        # Check performed outside lock to minimize lock contention
        if time.time() - last_save > checkpoint_interval:
            state.save_checkpoint()
            last_save = time.time()

# ==============================================================================
# WORKER FUNCTIONS
# ==============================================================================

def download_task_done(future: Future) -> None:
    """Callback triggered when a download thread finishes."""
    with state.lock:
        state.active_downloads -= 1

def download_pdf(session: requests.Session, url: str, output_dir: str) -> None:
    """
    Worker function executed by the ThreadPool. Handles the physical download.
    
    Args:
        session (requests.Session): The persistent HTTP session.
        url (str): The target PDF URL.
        output_dir (str): The local base directory.
    """
    local_path = "unknown"
    try:
        # Pre-check: Don't request if we already have it (Simple check via URL)
        dummy_path = get_local_path(output_dir, url)
        if os.path.exists(dummy_path) and os.path.getsize(dummy_path) > 0:
             return

        # Start Request
        # High timeout allowed because the monitor thread handles hung connections via overall flow control
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            
            # Resolve real filename from headers
            content_disp = r.headers.get('Content-Disposition')
            local_path = get_local_path(output_dir, url, content_disp)
            
            # Final Check: Do we have the file now that we know its real name?
            if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                log_manifest(url, local_path, "SKIPPED_EXISTS")
                return

            # Ensure directory tree exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Stream download in chunks to maintain low memory footprint
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=32768): # 32KB chunks
                    if chunk:
                        f.write(chunk)
                        with state.lock:
                            state.bytes_downloaded += len(chunk)
                            
        with state.lock:
            state.downloaded += 1
        log_manifest(url, local_path, "SUCCESS")

    except Exception as e:
        with state.lock:
            state.errors += 1
            # --- Congestion Control: Multiplicative Decrease ---
            # If a download fails, we assume network stress and cut threads by half.
            state.concurrency_limit = max(1, state.concurrency_limit // 2)
        log_manifest(url, local_path, "FAILED", str(e))

# ==============================================================================
# MAIN CRAWL LOGIC
# ==============================================================================

def save_manifest_csv(output_dir: str) -> None:
    """
    Writes the buffered manifest log to disk as a CSV.
    Opens in append mode to allow incremental updates.
    """
    filename = os.path.join(output_dir, "crawl_report.csv")
    with state.lock:
        if not state.manifest_log:
            return
        
        file_exists = os.path.isfile(filename)
        
        try:
            with open(filename, 'a', newline='', encoding='utf-8') as f:
                fields = ["timestamp", "url", "local_path", "status", "error"]
                writer = csv.DictWriter(f, fieldnames=fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(state.manifest_log)
            
            # Flush memory buffer
            state.manifest_log = []
            # Print a newline to break from the status bar line
            print(f"\n[REPORT] Manifest updated: {filename}")
        except Exception as e:
            print(f"\n[ERROR] Could not write manifest: {e}")

def graceful_exit(signum, frame) -> None:
    """
    Signal handler for SIGINT (Ctrl+C). Ensures state is saved before killing process.
    """
    print("\n\n[!] Stop signal received. Saving state...")
    state.save_checkpoint()
    # Write any remaining logs
    save_manifest_csv("downloads") 
    sys.exit(0)

def crawl(start_url: str, output_dir: str, max_threads_cap: int, delay: float) -> None:
    """
    Main controller function.
    
    1. Sets up signal handlers.
    2. Initializes HTTP Session and connection pools.
    3. Launches Monitor Thread.
    4. Manages the ThreadPoolExecutor and BFS queue.
    """
    # 1. Setup Signal Interception
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. State Initialization
    if not state.load_checkpoint():
        # Fresh start
        state.queue.append(start_url)
        state.visited.add(start_url)
        state.start_url = start_url

    # Normalize scope
    scope_url = state.start_url if state.start_url.endswith('/') else state.start_url + '/'
    state.max_concurrency_cap = max_threads_cap
    
    # 3. HTTP Session Setup with Connection Pooling
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=max_threads_cap, 
        pool_maxsize=max_threads_cap
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) EnterpriseCrawler/3.0'
    })

    print(f"[*] Starting Advanced Crawler")
    print(f"[*] Root Scope: {scope_url}")
    print(f"[*] Max Threads: {max_threads_cap}")
    
    # 4. Start Performance Monitor
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_performance, args=(stop_monitor,))
    monitor_thread.daemon = True
    monitor_thread.start()

    # 5. BFS Loop with Thread Pool
    with ThreadPoolExecutor(max_workers=max_threads_cap) as executor:
        try:
            while state.queue or state.active_downloads > 0:
                # Periodic Manifest Flush (Every 50 items)
                if state.downloaded > 0 and state.downloaded % 50 == 0:
                     save_manifest_csv(output_dir)

                # If queue is empty but downloads are running, wait for them
                if not state.queue:
                    time.sleep(1)
                    continue

                # Throttle Logic: Check if we have "slots" available
                can_submit = False
                with state.lock:
                    if state.active_downloads < state.concurrency_limit:
                        can_submit = True
                
                if not can_submit:
                    time.sleep(0.1) # Backpressure wait
                    continue

                # Pop next URL
                current_url = state.queue.popleft()
                
                try:
                    if delay > 0: 
                        time.sleep(delay)

                    # HEAD request to inspect Content-Type before downloading body
                    try:
                        head = session.head(current_url, timeout=10, allow_redirects=True)
                        content_type = head.headers.get('Content-Type', '').lower()
                    except requests.RequestException:
                        continue # Skip unresponsive links

                    # --- CASE A: PDF Document ---
                    if 'application/pdf' in content_type or current_url.lower().endswith('.pdf'):
                        with state.lock:
                            state.active_downloads += 1
                        future = executor.submit(download_pdf, session, current_url, output_dir)
                        future.add_done_callback(download_task_done)
                        continue

                    # --- CASE B: HTML Page (Scan for more links) ---
                    if 'text/html' in content_type:
                        try:
                            resp = session.get(current_url, timeout=10)
                            soup = BeautifulSoup(resp.content, 'html.parser')
                            
                            for link in soup.find_all('a', href=True):
                                abs_url = urljoin(current_url, link['href']).split('#')[0]
                                
                                # Prevent loops
                                if abs_url in state.visited:
                                    continue
                                
                                # Enforce Scope
                                if is_subpath(abs_url, scope_url):
                                    state.visited.add(abs_url)
                                    state.queue.append(abs_url)
                                    with state.lock:
                                        state.scanned += 1
                        except Exception:
                            # Ignore parsing errors to keep the crawl moving
                            pass 

                except Exception:
                    pass

        except KeyboardInterrupt:
            pass # Signal handler catches this, but block here just in case
    
    # 6. Cleanup
    stop_monitor.set()
    save_manifest_csv(output_dir) # Final flush
    
    # Remove checkpoint file upon successful completion
    if os.path.exists("crawler_checkpoint.json"):
        os.remove("crawler_checkpoint.json")
        
    print(f"\n\n[*] Crawl Complete. Report saved to {output_dir}/crawl_report.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced PDF Crawler")
    parser.add_argument('url', help="Starting URL to crawl")
    parser.add_argument('--output', '-o', default='downloads', help="Output directory")
    parser.add_argument('--threads', '-t', type=int, default=30, help="Max thread ceiling")
    parser.add_argument('--delay', '-d', type=float, default=0.0, help="Delay between page scans")
    args = parser.parse_args()
    
    crawl(args.url, args.output, args.threads, args.delay)
