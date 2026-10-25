"""
CLI: harvest catalogue-raisonné-scale corpora for contested masters from Wikidata/Commons.

Example:
    python scripts/harvest_wikidata_masters.py --artists vermeer rembrandt raphael goya velazquez bosch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition.wikidata_corpus import MASTER_QIDS, WikidataCorpusHarvester


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest master corpora from Wikidata + Wikimedia Commons")
    parser.add_argument("--artists", nargs="+", default=list(MASTER_QIDS), help="Artist keys to harvest")
    parser.add_argument("--limit-per-artist", type=int, default=50, help="Maximum paintings to harvest per master from Wikidata")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-dimension", type=int, default=4000)
    args = parser.parse_args()

    harvester = WikidataCorpusHarvester(concurrency=args.concurrency, max_dimension=args.max_dimension)
    for key in args.artists:
        try:
            harvester.harvest_artist(key, limit=args.limit_per_artist)
        except Exception as e:  # keep going with the next master
            print(f"[!] {key} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
