"""
CLI script to build and export the interactive ArtStylometry artwork browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.browser.renderer import ArtworkBrowserRenderer


def main():
    parser = argparse.ArgumentParser(description="Build interactive artwork browser and connoisseurship navigator")
    parser.add_argument(
        "--output",
        default="data/processed/browser.html",
        help="Path where browser.html will be exported (default: data/processed/browser.html)",
    )
    args = parser.parse_args()

    renderer = ArtworkBrowserRenderer()
    out_file = renderer.export(args.output)
    print(f"[OK] Successfully compiled ArtStylometry Visual Browser to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
