"""
ArtStylometry - High-Performance Local HTTP Server with Auto-Port Selection & Browser Launch.
"""

from __future__ import annotations

import functools
import http.server
import socket
import sys
import webbrowser
from pathlib import Path


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(preferred_ports: list[int] = [8000, 8080, 8088, 8888, 9000]) -> int:
    for port in preferred_ports:
        if is_port_free(port):
            return port
    # Fallback to random free port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main() -> None:
    # Ensure UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    root_dir = Path(__file__).resolve().parent.parent
    port = find_available_port()
    url = f"http://localhost:{port}/"

    print("=" * 75)
    print("      🎨 ARTSTYLOMETRY — COMPUTATIONAL STYLOMETRY & CONNOISSEURSHIP")
    print("=" * 75)
    print(f" [+] Root Directory       : {root_dir}")
    print(f" [+] HTTP Server Active on: {url}")
    print(f" [+] Launching default web browser...")
    print(f" [!] Press Ctrl+C in this console to stop the server.")
    print("=" * 75)

    handler_class = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root_dir))
    
    # Open browser
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f" [!] Browser launch notification: {e}")

    try:
        with http.server.ThreadingHTTPServer(("0.0.0.0", port), handler_class) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n [✓] ArtStylometry server stopped successfully. Goodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
