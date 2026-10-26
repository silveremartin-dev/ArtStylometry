"""
Automated Algorithmic Corpus Deduplication Pipeline.
Performs 100% automated computer vision deduplication across all 12 master ateliers
using 256-bit perceptual difference hashing (dHash), 3D color structure histograms,
geometric ORB/RANSAC keypoint matching, and institutional provenance verification.
Retains the highest-resolution, sharpest image for each physical artwork cluster.
Zero manual hardcoded ID lists or title rules.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.deduplicator import AlgorithmicArtworkDeduplicator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deduplicate_corpus")


def load_raw_baseline_records() -> List[Dict[str, Any]]:
    """Loads the complete unadulterated 1,137 records baseline from git revision 03585ee."""
    raw_str = subprocess.check_output(
        ["git", "show", "03585ee:data/processed/contested_masters_audit.json"],
        encoding="utf-8"
    )
    records: List[Dict[str, Any]] = json.loads(raw_str)
    return records


def run_algorithmic_deduplication() -> Tuple[int, int]:
    raw_records = load_raw_baseline_records()
    logger.info("Loaded %d raw baseline records across 12 master ateliers.", len(raw_records))

    # Group by canonical artist folder
    records_by_artist = defaultdict(list)
    for r in raw_records:
        lp = r.get("local_path", "")
        # Standardize local_path to existing disk file
        adir = Path(lp).parent.name.lower()
        records_by_artist[adir].append(r)

    deduplicator = AlgorithmicArtworkDeduplicator(
        hamming_threshold=0.16,
        color_threshold=0.88,
        aspect_ratio_tolerance=0.18,
        ransac_inlier_threshold=18,
    )

    masters_dir = Path("data/raw/contested_masters")
    total_before = 0
    total_after = 0

    for artist_dir in sorted(masters_dir.iterdir()):
        if not artist_dir.is_dir():
            continue
        adir_name = artist_dir.name.lower()
        artist_recs = records_by_artist.get(adir_name, [])
        if not artist_recs:
            continue

        total_before += len(artist_recs)
        kept, pruned = deduplicator.deduplicate_records(artist_recs)
        total_after += len(kept)

        # Write clean catalog.json
        cat_file = artist_dir / "catalog.json"
        with open(cat_file, "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2, ensure_ascii=False)

        logger.info(
            "[%s] Deduplicated: %d -> %d artworks (%d duplicates pruned by vision & provenance)",
            artist_dir.name, len(artist_recs), len(kept), len(pruned)
        )
        for p in pruned:
            logger.info("   - [%s] Pruned %s (%s) in favor of %s (%s): %s",
                        artist_dir.name, p["duplicate_id"], p["duplicate_res"],
                        p["retained_id"], p["retained_res"], p["title"])

    logger.info("Total Algorithmic Deduplication: %d -> %d artworks (%d duplicates pruned)",
                total_before, total_after, total_before - total_after)
    return total_before, total_after


if __name__ == "__main__":
    run_algorithmic_deduplication()
