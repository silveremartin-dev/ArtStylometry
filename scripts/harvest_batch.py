"""
CLI Script for High-Throughput Batch Ingestion of Museum Paintings.
Harvests up to 1,000 fine art paintings at high resolution (up to 4,000px)
with live terminal progress and automatic resume support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition.batch_harvester import BatchHarvester


def main():
    parser = argparse.ArgumentParser(description="Harvest museum fine art paintings in batch at up to 4000px")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of paintings to ingest (default: 1000)",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=4000,
        help="Maximum width or height in pixels (default: 4000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)",
    )
    parser.add_argument(
        "--source",
        choices=["all", "cma", "nga", "met", "wikidata", "extended"],
        default="all",
        help="Source: 'all' (NGA + CMA), 'extended' (Met + Wikidata museums), or a single source",
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=None,
        help="Target storage quota in GB (e.g. 30.0 or 60.0). Harvester pauses once reached.",
    )
    parser.add_argument(
        "--dest-dir",
        default="data/raw/paintings",
        help="Target folder for raw painting JPEGs (default: data/raw/paintings)",
    )
    parser.add_argument(
        "--catalog",
        default="data/raw/paintings_catalog.json",
        help="Path to save metadata catalog (default: data/raw/paintings_catalog.json)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("ArtStylometry High-Throughput Batch Painting Harvester")
    print("=" * 70)
    print(f"Source          : {args.source.upper()}")
    print(f"Target count    : {args.limit} paintings")
    if args.target_gb:
        print(f"Target disk cap : {args.target_gb:.1f} GB")
    print(f"Max resolution  : {args.max_dimension} px")
    print(f"Concurrency     : {args.concurrency} workers")
    print(f"Destination dir : {Path(args.dest_dir).resolve()}")
    print(f"Catalog file    : {Path(args.catalog).resolve()}")
    print("=" * 70)

    harvester = BatchHarvester(
        dest_dir=args.dest_dir,
        catalog_path=args.catalog,
        max_dimension=args.max_dimension,
        concurrency=args.concurrency,
    )

    records = harvester.harvest_batch(
        total_limit=args.limit,
        source=args.source,
        target_gb=args.target_gb,
    )

    total_bytes = sum(r.filesize_bytes for r in records)
    total_gb = total_bytes / (1024 ** 3)
    print("-" * 70)
    print(f"Ingestion Summary:")
    print(f"  Total Paintings Harvested : {len(records)}")
    print(f"  Total Disk Space Consumed : {total_gb:.2f} GB ({total_bytes / (1024 ** 2):.1f} MB)")
    if records:
        avg_mb = (total_bytes / len(records)) / (1024 ** 2)
        print(f"  Average Size per Painting : {avg_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
