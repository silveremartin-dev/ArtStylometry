"""
Attribution Heatmap Generator CLI.
Generates intra-canvas spatial attribution maps on an artwork given reference works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ArtStylometryPipeline


def main():
    parser = argparse.ArgumentParser(description="Generate spatial attribution heatmap on artwork")
    parser.add_argument("--artwork", required=True, help="Path to test painting image")
    parser.add_argument("--references", required=True, nargs="+", help="Paths to canonical reference paintings")
    parser.add_argument("--artist", default="Master", help="Name of reference master artist")
    parser.add_argument("--out-dir", default="data/processed/audits", help="Output directory")
    parser.add_argument("--deep", action="store_true", help="Attempt to use DINOv2 deep vision transformer")

    args = parser.parse_args()

    pipeline = ArtStylometryPipeline(use_deep_model=args.deep)
    profile = pipeline.build_artist_profile(args.artist, args.references)
    report_path = pipeline.audit_artwork(args.artwork, profile, args.out_dir)

    print(f"[OK] Audit complete! HTML report generated at: {report_path}")


if __name__ == "__main__":
    main()
