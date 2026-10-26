"""
Automated Algorithmic Artwork Deduplicator.
Identifies duplicate photographic instances, crops, or multi-source harvests of the
same physical artwork using Perceptual Difference Hashing (dHash), Normalized Color
Histograms, Geometric ORB/RANSAC Feature Matching, and Institutional Provenance Verification.
Fully automated with zero manual hardcoded artist lists.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class VisualFingerprint:
    """Compact perceptual and geometric fingerprint of an artwork image."""
    artwork_id: str
    local_path: str
    width: int
    height: int
    aspect_ratio: float
    dhash: np.ndarray  # 256 boolean bits
    color_hist: np.ndarray  # 64-bin normalized color histogram
    sharpness: float  # Variance of Laplacian
    keypoints: Optional[List[Any]] = None
    descriptors: Optional[np.ndarray] = None


class AlgorithmicArtworkDeduplicator:
    """100% automated perceptual and institutional image deduplicator for fine art corpora."""

    def __init__(
        self,
        hamming_threshold: float = 0.16,
        color_threshold: float = 0.88,
        aspect_ratio_tolerance: float = 0.15,
        ransac_inlier_threshold: int = 18,
    ) -> None:
        self.hamming_threshold = hamming_threshold
        self.color_threshold = color_threshold
        self.aspect_ratio_tolerance = aspect_ratio_tolerance
        self.ransac_inlier_threshold = ransac_inlier_threshold
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @staticmethod
    def normalize_str(s: str) -> str:
        """Strips punctuation and whitespace for normalized comparison."""
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    @staticmethod
    def extract_canonical_museum(museum_str: str) -> str:
        """
        Normalizes institutional repository names to prevent conflating distinct physical
        holdings across different world museums (e.g. Louvre vs Prado).
        """
        s = AlgorithmicArtworkDeduplicator.normalize_str(museum_str)
        if not s or s.startswith("http") or s == "unknown":
            return "Unknown"

        known_institutions = {
            "louvre": "Musée du Louvre",
            "uffizi": "Uffizi Gallery",
            "prado": "Museo del Prado",
            "nationalgallery": "National Gallery, London",
            "hermitage": "Hermitage Museum",
            "metropolitan": "Metropolitan Museum of Art",
            "ambrosiana": "Pinacoteca Ambrosiana",
            "brera": "Pinacoteca di Brera",
            "ashmolean": "Ashmolean Museum",
            "basel": "Kunstmuseum Basel",
            "gemaldegalerie": "Gemäldegalerie Berlin",
            "royalcollection": "Royal Collection",
            "czartoryski": "Czartoryski Museum",
            "vatican": "Pinacoteca Vaticana",
            "scotland": "National Galleries of Scotland",
            "cleveland": "Cleveland Museum of Art",
            "hammer": "Hammer Museum",
            "detroit": "Detroit Institute of Arts",
            "bavarian": "Bavarian State Painting Collections",
            "altepinakothek": "Bavarian State Painting Collections",
            "rijksmuseum": "Rijksmuseum",
            "mauritshuis": "Mauritshuis",
            "thyssen": "Thyssen-Bornemisza Museum",
            "kunsthistorisches": "Kunsthistorisches Museum",
            "nationalgalleryofart": "National Gallery of Art, Washington D.C.",
        }
        for k, v in known_institutions.items():
            if k in s:
                return v
        return museum_str.strip()

    @staticmethod
    def are_museums_compatible(m1: str, m2: str) -> bool:
        """
        Determines if two records can be photographic duplicates of the same physical object.
        If both records belong to distinct accredited world museums (e.g. Louvre vs Prado),
        they are distinct physical canvases and must not be merged.
        """
        cm1 = AlgorithmicArtworkDeduplicator.extract_canonical_museum(m1)
        cm2 = AlgorithmicArtworkDeduplicator.extract_canonical_museum(m2)
        if cm1 == "Unknown" or cm2 == "Unknown":
            return True
        return cm1 == cm2

    def compute_fingerprint(self, image_path: str | Path, artwork_id: str) -> Optional[VisualFingerprint]:
        """Computes dHash, color histogram, sharpness, and ORB keypoints."""
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

                # 4. ORB keypoints on bounded grayscale array (max 800px)
                cv_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                kp, des = None, None
                if cv_img is not None:
                    ch, cw = cv_img.shape
                    if max(ch, cw) > 800:
                        scale = 800.0 / max(ch, cw)
                        cv_small = cv2.resize(cv_img, (int(cw * scale), int(ch * scale)))
                    else:
                        cv_small = cv_img
                    kp, des = self._orb.detectAndCompute(cv_small, None)

                return VisualFingerprint(
                    artwork_id=artwork_id,
                    local_path=str(p),
                    width=w,
                    height=h,
                    aspect_ratio=w / float(h),
                    dhash=dhash,
                    color_hist=hist,
                    sharpness=sharpness,
                    keypoints=kp,
                    descriptors=des,
                )
        except Exception as e:
            logger.warning("Failed computing fingerprint for %s: %s", image_path, e)
            return None

    def are_duplicates(
        self,
        f1: VisualFingerprint,
        f2: VisualFingerprint,
        rec1: Optional[Dict[str, Any]] = None,
        rec2: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, float]:
        """
        Evaluates whether two records correspond to the same physical artwork:
        1. Exact digital clone (almost identical pixels regardless of provenance).
        2. Same museum provenance + exact normalized title.
        3. Same museum provenance + strong perceptual dHash match.
        4. Same museum provenance + high geometric RANSAC homography inliers (crops/details).
        """
        # Fractional Hamming distance (0 = identical, 1 = completely inverted)
        hamming = float(np.mean(f1.dhash != f2.dhash))
        color_sim = float(np.dot(f1.color_hist, f2.color_hist))
        ar_ratio = min(f1.aspect_ratio, f2.aspect_ratio) / max(f1.aspect_ratio, f2.aspect_ratio)

        # Rule 1: Exact digital clone (identical scan files)
        if hamming <= 0.05 and ar_ratio >= 0.98:
            return True, hamming, color_sim

        m1 = rec1.get("museum", "") if rec1 else ""
        m2 = rec2.get("museum", "") if rec2 else ""

        # Enforce institutional preservation: cannot merge distinct accredited world museums
        if not self.are_museums_compatible(m1, m2):
            return False, hamming, color_sim

        # Rule 2: Institutional provenance identity + exact normalized title
        t1 = self.normalize_str(rec1.get("title", "")) if rec1 else ""
        t2 = self.normalize_str(rec2.get("title", "")) if rec2 else ""
        if t1 and t2 and t1 == t2:
            return True, hamming, color_sim

        # Rule 3: Institutional compatibility + perceptual dHash match
        if ar_ratio >= (1.0 - self.aspect_ratio_tolerance) and hamming <= self.hamming_threshold:
            return True, hamming, color_sim

        # Rule 4: Institutional compatibility + geometric RANSAC match (crops / frame differences)
        # Fast candidate gate: only run keypoint matching if images exhibit lexical or strong structural candidate affinity
        stop_words = {"the", "and", "with", "from", "version", "oil", "panel", "canvas", "portrait"}
        raw_t1 = rec1.get("title", "").lower() if rec1 else ""
        raw_t2 = rec2.get("title", "").lower() if rec2 else ""
        tok1 = set(w.rstrip("s") for w in re.findall(r"[a-z0-9]+", raw_t1) if len(w) >= 3 and w not in stop_words)
        tok2 = set(w.rstrip("s") for w in re.findall(r"[a-z0-9]+", raw_t2) if len(w) >= 3 and w not in stop_words)
        # If titles share meaningful keywords OR global perceptual dHash indicates near-duplicate framing:
        is_candidate = (hamming <= 0.22) or bool(tok1 & tok2)
        if not is_candidate:
            return False, hamming, color_sim

        if f1.descriptors is not None and f2.descriptors is not None:
            matches = self._bf.match(f1.descriptors, f2.descriptors)
            good = [m for m in matches if m.distance < 50]
            if len(good) >= 15:
                src_pts = np.float32([f1.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([f2.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = int(np.sum(mask)) if mask is not None else 0
                if inliers >= self.ransac_inlier_threshold:
                    return True, hamming, color_sim

        return False, hamming, color_sim

    def deduplicate_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Takes raw catalog records, computes fingerprints, finds duplicate clusters,
        and automatically retains only the highest-resolution representative record.
        Enriches the retained record with complete metadata from merged duplicates.
        Guarantees that distinct accredited museums can never be transitively merged.
        Returns (kept_records, pruned_records).
        """
        # Step 1: Compute fingerprints for all available records
        fingerprints: Dict[str, VisualFingerprint] = {}
        records_by_id = {r["artwork_id"]: r for r in records}

        for r in records:
            aid = r["artwork_id"]
            lp = r.get("local_path", "")
            fp = self.compute_fingerprint(lp, aid)
            if fp is not None:
                fingerprints[aid] = fp

        # Step 2: Pairwise comparison and union-find clustering with institutional preservation
        parent: Dict[str, str] = {aid: aid for aid in fingerprints}
        canonical_museums: Dict[str, str] = {
            aid: self.extract_canonical_museum(records_by_id[aid].get("museum", ""))
            for aid in fingerprints
        }

        def find(u: str) -> str:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def union(u: str, v: str) -> bool:
            ru = find(u)
            rv = find(v)
            if ru == rv:
                return True
            mu = canonical_museums[ru]
            mv = canonical_museums[rv]
            # Strict institutional conflict: cannot merge two distinct accredited museums (e.g. Louvre vs Prado)
            if mu != "Unknown" and mv != "Unknown" and mu != mv:
                return False
            parent[ru] = rv
            if mv == "Unknown" and mu != "Unknown":
                canonical_museums[rv] = mu
            return True

        fp_list = list(fingerprints.values())
        n = len(fp_list)
        for i in range(n):
            f1 = fp_list[i]
            r1 = records_by_id[f1.artwork_id]
            for j in range(i + 1, n):
                f2 = fp_list[j]
                r2 = records_by_id[f2.artwork_id]
                is_dup, ham, c_sim = self.are_duplicates(f1, f2, r1, r2)
                if is_dup:
                    merged = union(f1.artwork_id, f2.artwork_id)
                    if merged:
                        logger.info(
                            "Algorithmic duplicate detected: '%s' <-> '%s' (Hamming: %.3f, ColorSim: %.3f)",
                            f1.artwork_id, f2.artwork_id, ham, c_sim
                        )

        # Step 3: Group into clusters
        clusters: Dict[str, List[str]] = {}
        for aid in fingerprints:
            root = find(aid)
            clusters.setdefault(root, []).append(aid)

        # Step 4: For each cluster, pick the highest-resolution representative record
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
                winner_rec = dict(records_by_id[winner_id])

                # Enrich metadata from losers if winner had generic or missing fields
                for loser_id in sorted_ids[1:]:
                    loser_rec = records_by_id[loser_id]
                    for field in ["creation_date", "medium", "dimensions", "museum", "official_attribution"]:
                        if (not winner_rec.get(field) or winner_rec.get(field) in ["Unknown", ""]) and loser_rec.get(field) and loser_rec.get(field) != "Unknown":
                            winner_rec[field] = loser_rec[field]

                    pruned_records.append({
                        "duplicate_id": loser_id,
                        "retained_id": winner_id,
                        "duplicate_res": f"{fingerprints[loser_id].width}x{fingerprints[loser_id].height}",
                        "retained_res": f"{fingerprints[winner_id].width}x{fingerprints[winner_id].height}",
                        "title": records_by_id[loser_id].get("title", ""),
                    })
                kept_records.append(winner_rec)

        # Records whose images could not be fingerprinted are kept untouched
        for r in records:
            if r["artwork_id"] not in fingerprints:
                kept_records.append(r)

        return kept_records, pruned_records
