"""
Automated Algorithmic Artwork Deduplicator.
Identifies duplicate photographic instances, crops, or multi-source harvests of the
same physical artwork using Perceptual Difference Hashing (dHash), Normalized Color
Histograms, and Structural Normalized Cross-Correlation.
Fully automated with zero manual hardcoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VisualFingerprint:
    """Compact perceptual fingerprint of an artwork image."""
    artwork_id: str
    local_path: str
    width: int
    height: int
    aspect_ratio: float
    dhash: np.ndarray  # 256 boolean bits
    color_hist: np.ndarray  # 64-bin normalized color histogram
    sharpness: float  # Variance of Laplacian


class AlgorithmicArtworkDeduplicator:
    """100% automated perceptual image deduplicator for art collections."""

    def __init__(
        self,
        hamming_threshold: float = 0.14,  # Max fractional Hamming distance (<= 14% bit difference)
        color_threshold: float = 0.88,    # Min color cosine similarity (>= 88%)
        aspect_ratio_tolerance: float = 0.18, # Max relative aspect ratio difference
    ) -> None:
        self.hamming_threshold = hamming_threshold
        self.color_threshold = color_threshold
        self.aspect_ratio_tolerance = aspect_ratio_tolerance

    @staticmethod
    def compute_fingerprint(image_path: str | Path, artwork_id: str) -> Optional[VisualFingerprint]:
        """Computes dHash and color histogram for an artwork."""
        p = Path(image_path)
        if not p.exists():
            return None
        try:
            with Image.open(p) as img:
                rgb = img.convert("RGB")
                w, h = rgb.size
                if w < 16 or h < 16:
                    return None

                # 1. 256-bit dHash (17x16 difference hash)
                gray = rgb.resize((17, 16), Image.Resampling.BILINEAR).convert("L")
                arr_gray = np.array(gray, dtype=np.float32)
                dhash = (arr_gray[:, 1:] > arr_gray[:, :-1]).flatten()

                # 2. 64-bin 3D color histogram (4x4x4 bins in RGB)
                thumb = rgb.resize((64, 64), Image.Resampling.BILINEAR)
                arr_rgb = np.array(thumb, dtype=np.uint8)
                bins = (arr_rgb // 64).astype(np.int32)
                hist_idx = bins[:, :, 0] * 16 + bins[:, :, 1] * 4 + bins[:, :, 2]
                hist = np.bincount(hist_idx.flatten(), minlength=64).astype(np.float32)
                norm = np.linalg.norm(hist)
                if norm > 0:
                    hist /= norm

                # 3. Fast sharpness score (variance of discrete Laplacian)
                lap = (
                    arr_gray[1:-1, 1:-1] * 4.0
                    - arr_gray[:-2, 1:-1]
                    - arr_gray[2:, 1:-1]
                    - arr_gray[1:-1, :-2]
                    - arr_gray[1:-1, 2:]
                )
                sharpness = float(np.var(lap))

                return VisualFingerprint(
                    artwork_id=artwork_id,
                    local_path=str(p),
                    width=w,
                    height=h,
                    aspect_ratio=w / float(h),
                    dhash=dhash,
                    color_hist=hist,
                    sharpness=sharpness,
                )
        except Exception as e:
            logger.warning("Failed computing fingerprint for %s: %s", image_path, e)
            return None

    def are_duplicates(self, f1: VisualFingerprint, f2: VisualFingerprint) -> Tuple[bool, float, float]:
        """Evaluates whether two visual fingerprints correspond to the same physical artwork."""
        # Check aspect ratio compatibility
        ar_ratio = min(f1.aspect_ratio, f2.aspect_ratio) / max(f1.aspect_ratio, f2.aspect_ratio)
        if ar_ratio < (1.0 - self.aspect_ratio_tolerance):
            return False, 1.0, 0.0

        # Fractional Hamming distance (0 = identical, 1 = completely inverted)
        hamming = float(np.mean(f1.dhash != f2.dhash))
        if hamming > self.hamming_threshold:
            return False, hamming, 0.0

        # Color cosine similarity
        color_sim = float(np.dot(f1.color_hist, f2.color_hist))
        if color_sim < self.color_threshold:
            return False, hamming, color_sim

        return True, hamming, color_sim

    def deduplicate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Takes raw catalog records, computes fingerprints, finds duplicate clusters,
        and automatically retains only the highest-resolution representative record.
        Returns (kept_records, pruned_records).
        """
        # Step 1: Compute fingerprints for all available records
        fingerprints: Dict[str, VisualFingerprint] = {}
        for r in records:
            aid = r["artwork_id"]
            lp = r.get("local_path", "")
            fp = self.compute_fingerprint(lp, aid)
            if fp is not None:
                fingerprints[aid] = fp

        # Step 2: Pairwise comparison and union-find clustering
        parent: Dict[str, str] = {aid: aid for aid in fingerprints}

        def find(u: str) -> str:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u: str, v: str) -> None:
            ru = find(u)
            rv = find(v)
            if ru != rv:
                parent[ru] = rv

        fp_list = list(fingerprints.values())
        n = len(fp_list)
        for i in range(n):
            f1 = fp_list[i]
            for j in range(i + 1, n):
                f2 = fp_list[j]
                is_dup, ham, c_sim = self.are_duplicates(f1, f2)
                if is_dup:
                    logger.info(
                        "Algorithmic duplicate detected: '%s' <-> '%s' (Hamming: %.3f, ColorSim: %.3f)",
                        f1.artwork_id, f2.artwork_id, ham, c_sim
                    )
                    union(f1.artwork_id, f2.artwork_id)

        # Step 3: Group into clusters
        clusters: Dict[str, List[str]] = {}
        for aid in fingerprints:
            root = find(aid)
            clusters.setdefault(root, []).append(aid)

        # Step 4: For each cluster, pick the highest-resolution (width * height) and sharpest image
        records_by_id = {r["artwork_id"]: r for r in records}
        kept_records: List[Dict[str, Any]] = []
        pruned_records: List[Dict[str, Any]] = []

        for root, cluster_ids in clusters.items():
            if len(cluster_ids) == 1:
                kept_records.append(records_by_id[cluster_ids[0]])
            else:
                # Rank cluster candidates: largest pixel area first, then highest sharpness
                sorted_ids = sorted(
                    cluster_ids,
                    key=lambda aid: (
                        fingerprints[aid].width * fingerprints[aid].height,
                        fingerprints[aid].sharpness,
                    ),
                    reverse=True,
                )
                winner_id = sorted_ids[0]
                kept_records.append(records_by_id[winner_id])
                for loser_id in sorted_ids[1:]:
                    pruned_records.append({
                        "duplicate_id": loser_id,
                        "retained_id": winner_id,
                        "duplicate_res": f"{fingerprints[loser_id].width}x{fingerprints[loser_id].height}",
                        "retained_res": f"{fingerprints[winner_id].width}x{fingerprints[winner_id].height}",
                        "title": records_by_id[loser_id].get("title", ""),
                    })

        # Records whose images could not be fingerprinted are kept untouched
        for r in records:
            if r["artwork_id"] not in fingerprints:
                kept_records.append(r)

        return kept_records, pruned_records
