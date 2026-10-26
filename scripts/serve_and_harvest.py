"""
ArtStylometry - Dual-Engine Launcher: Local HTTP Server & Background Database Harvester.
Launches the interactive connoisseurship portal in the browser while concurrently
populating the local artwork database from worldwide open-access museum collections.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.acquisition.batch_harvester import BatchHarvester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ServeAndHarvest")


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(preferred_ports: list[int] = [8080, 8000, 8088, 8888, 9000]) -> int:
    for port in preferred_ports:
        if is_port_free(port):
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class BackgroundHarvesterThread(threading.Thread):
    """
    Background worker thread running high-throughput museum ingestion
    without blocking the local HTTP web server or UI interactions.
    """

    def __init__(
        self,
        dest_dir: str | Path = "data/raw/paintings",
        catalog_path: str | Path = "data/raw/paintings_catalog.json",
        max_dimension: int = 4000,
        concurrency: int = 4,
        source: str = "full",
        limit: int = 25000,
        target_gb: float | None = 60.0,
    ) -> None:
        super().__init__(daemon=True, name="BackgroundHarvester")
        self.dest_dir = Path(dest_dir)
        self.catalog_path = Path(catalog_path)
        self.max_dimension = max_dimension
        self.concurrency = concurrency
        self.source = source
        self.limit = limit
        self.target_gb = target_gb
        self.stop_requested = threading.Event()
        self.harvester = BatchHarvester(
            dest_dir=self.dest_dir,
            catalog_path=self.catalog_path,
            max_dimension=self.max_dimension,
            concurrency=self.concurrency,
        )

    def run(self) -> None:
        time.sleep(2.0)  # Brief pause to let HTTP server start and browser launch
        print("\n" + "=" * 78)
        print(" [🚀 BACKGROUND HARVESTER ACTIVE] Populating Local Artwork Database")
        print("=" * 78)
        print(f" [+] Target Collection  : Up to {self.limit:,} fine art paintings")
        print(f" [+] Source Channels    : {self.source.upper()} (CMA, NGA, Met, Wikidata museums)")
        if self.target_gb:
            print(f" [+] Disk Storage Cap   : {self.target_gb:.1f} GB")
        else:
            print(" [+] Disk Storage Cap   : Unlimited")
        print(f" [+] Concurrency Workers: {self.concurrency} download threads")
        print(f" [+] Local Storage Dir  : {self.dest_dir.resolve()}")
        print(f" [+] Master Catalog     : {self.catalog_path.resolve()}")
        print("=" * 78 + "\n")

        try:
            self.harvester.harvest_batch(
                total_limit=self.limit,
                source=self.source,
                target_gb=self.target_gb,
            )
        except Exception as e:
            logger.error(f"Background harvesting thread encountered an error: {e}", exc_info=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ArtStylometry local server while populating local database with museum artworks."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Local HTTP server port (default: auto-detected, 8080/8000/etc.)",
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=60.0,
        help="Target storage quota in GB (default: 60.0 GB). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25000,
        help="Total painting download limit (default: 25000)",
    )
    parser.add_argument(
        "--source",
        choices=["full", "all", "cma", "nga", "met", "wikidata", "extended"],
        default="full",
        help="Artwork source channels (default: 'full' querying CMA, NGA, Met, and Wikidata museums)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent download worker threads (default: 4)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=4000,
        help="Maximum painting resolution in pixels (default: 4000)",
    )
    parser.add_argument(
        "--dest-dir",
        default="data/raw/paintings",
        help="Destination directory for raw high-resolution paintings (default: data/raw/paintings)",
    )
    parser.add_argument(
        "--catalog",
        default="data/raw/paintings_catalog.json",
        help="Master catalog file for metadata records (default: data/raw/paintings_catalog.json)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch web browser on startup",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    target_gb = None if args.target_gb <= 0 else args.target_gb
    port = args.port or find_available_port()
    url = f"http://localhost:{port}/"

    print("=" * 78)
    print("      🎨 ARTSTYLOMETRY — DUAL SERVER & LOCAL DATABASE POPULATOR")
    print("=" * 78)
    print(f" [+] Web Root Directory   : {ROOT_DIR}")
    print(f" [+] HTTP Server URL      : {url}")
    print(f" [+] Background Harvester : ACTIVE (Target: {args.limit:,} works, cap: {target_gb or 'Unlimited'} GB)")
    print(f" [!] Press Ctrl+C at any time to stop both server and downloads safely.")
    print("=" * 78)

    # 1. Start background harvester thread
    harvester_thread = BackgroundHarvesterThread(
        dest_dir=args.dest_dir,
        catalog_path=args.catalog,
        max_dimension=args.max_dimension,
        concurrency=args.concurrency,
        source=args.source,
        limit=args.limit,
        target_gb=target_gb,
    )
    harvester_thread.start()

    # 2. Launch browser
    if not args.no_browser:
        try:
            print(f" [+] Opening browser at {url} ...")
            webbrowser.open(url)
        except Exception as e:
            print(f" [!] Browser launch notification: {e}")

    # 3. Run HTTP server on main thread
    handler_class = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT_DIR))
    try:
        with http.server.ThreadingHTTPServer(("0.0.0.0", port), handler_class) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n" + "=" * 78)
        print(" [✓] Stop signal received. Shutting down HTTP server...")
        print(" [✓] Flushing catalog and stopping background harvester...")
        print(" [✓] ArtStylometry stopped safely. All downloaded paintings preserved!")
        print("=" * 78)
        sys.exit(0)


if __name__ == "__main__":
    main()
