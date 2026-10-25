"""
Near-duplicate painting detection via perceptual difference hashing (dHash).

The same canvas is frequently delivered by several sources (e.g. NGA IIIF and
Wikimedia Commons). Duplicates inside a master's reference set would shrink the
covariance artificially and invalidate leave-one-canvas-out statistics, so they
are collapsed before any audit. The highest-resolution copy is kept; museum-native
sources are preferred over Commons mirrors at equal resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def dhash(image_path: str | Path, hash_size: int = 8) -> Tuple[int, float]:
    """Returns (64-bit difference hash, aspect ratio) for an image file."""
    with Image.open(image_path) as im:
        im.draft("L", (256, 256))  # fast JPEG DCT-domain downscale
        aspect = im.width / max(1, im.height)
        g = np.asarray(im.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value, aspect


def _source_priority(artwork_id: str) -> int:
    return 1 if artwork_id.startswith("wd_") else 0


def deduplicate_records(
    records: List[Dict[str, Any]],
    max_hamming: int = 8,
    max_aspect_diff: float = 0.04,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    """
    Collapses near-duplicate paintings.

    Returns:
        (kept_records, removed_pairs) where removed_pairs lists (removed_id, kept_id).
    """
    hashed = []
    for r in records:
        try:
            h, a = dhash(r["local_path"])
        except Exception:
            h, a = None, None
        hashed.append((r, h, a))

    # Best copies first: native museum source, then highest pixel count
    order = sorted(
        range(len(hashed)),
        key=lambda i: (_source_priority(hashed[i][0]["artwork_id"]),
                       -int(hashed[i][0].get("width", 0)) * int(hashed[i][0].get("height", 0))),
    )
    kept_idx: List[int] = []
    removed: List[Tuple[str, str]] = []
    for i in order:
        r, h, a = hashed[i]
        dup_of = None
        if h is not None:
            for k in kept_idx:
                _, hk, ak = hashed[k]
                if hk is None:
                    continue
                if abs(a - ak) / max(ak, 1e-6) <= max_aspect_diff and bin(h ^ hk).count("1") <= max_hamming:
                    dup_of = hashed[k][0]["artwork_id"]
                    break
        if dup_of:
            removed.append((r["artwork_id"], dup_of))
        else:
            kept_idx.append(i)
    kept_set = set(kept_idx)
    return [hashed[i][0] for i in range(len(hashed)) if i in kept_set], removed
