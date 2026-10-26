"""
Large-Scale Collection Manifold Clustering & 3D WebGL Point-Cloud Generator.
Scales the stylometric manifold to thousands of fine art paintings across 12 master ateliers
and worldwide museum collections (Cleveland, NGA, Met, Louvre, Rijksmuseum, Wikidata).
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.micro_extractor import MicroFactureExtractor
from src.preprocessing.tiler import ArtworkTiler
from src.visualization.manifold_explorer import export_manifold_explorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scale_manifold")


def parse_century(date_str: str) -> Optional[int]:
    m = re.search(r"(1[2-9]\d\d|20\d\d)", str(date_str))
    return int(m.group(1)) // 100 + 1 if m else 0


def load_all_records() -> List[Dict[str, Any]]:
    records = []
    seen_ids = set()

    # 1. Global gallery catalog (12,346 paintings)
    gal_path = Path("data/processed/global_gallery_catalog.json")
    if gal_path.exists():
        gallery = json.loads(gal_path.read_text(encoding="utf-8"))
        for g in gallery:
            aid = g.get("id")
            lp = g.get("path", "")
            if aid and aid not in seen_ids and Path(lp).exists():
                seen_ids.add(aid)
                records.append({
                    "artwork_id": aid,
                    "title": g.get("title", "Untitled"),
                    "artist": g.get("artist", "Unknown Master"),
                    "creation_date": str(g.get("date", "")),
                    "century": parse_century(str(g.get("date", ""))),
                    "museum": g.get("museum", "Museum Collection"),
                    "local_path": lp,
                    "tag": g.get("tag", "GALLERY")
                })

    # 2. Contested masters (1,137 paintings)
    masters_path = Path("data/processed/contested_masters_audit.json")
    if masters_path.exists():
        masters = json.loads(masters_path.read_text(encoding="utf-8"))
        for m in masters:
            aid = m.get("artwork_id")
            lp = m.get("local_path", "")
            if aid and aid not in seen_ids and Path(lp).exists():
                seen_ids.add(aid)
                records.append({
                    "artwork_id": aid,
                    "title": m.get("title", "Untitled"),
                    "artist": m.get("artist_canonical", "Unknown Master"),
                    "creation_date": str(m.get("creation_date", "")),
                    "century": parse_century(str(m.get("creation_date", ""))),
                    "museum": m.get("museum", "Museum Collection"),
                    "local_path": lp,
                    "tag": "MASTER_CORPUS"
                })

    # 3. General catalog (Cleveland, NGA, Met)
    cat_path = Path("data/raw/paintings_catalog.json")
    if cat_path.exists():
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
        for c in cat:
            aid = c.get("artwork_id")
            lp = c.get("local_path", "")
            if aid and aid not in seen_ids and Path(lp).exists():
                seen_ids.add(aid)
                records.append({
                    "artwork_id": aid,
                    "title": c.get("title", "Untitled"),
                    "artist": c.get("artist", "Unknown Artist"),
                    "creation_date": str(c.get("creation_date", "")),
                    "century": parse_century(str(c.get("creation_date", ""))),
                    "museum": c.get("museum", "Museum Collection"),
                    "local_path": lp,
                    "tag": "HARVESTED"
                })

    return records


# Global extractor to avoid re-instantiation overhead on every image
_EXTRACTOR = None


def get_extractor() -> MicroFactureExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = MicroFactureExtractor(use_deep_model=False, embedding_dim=128)
    return _EXTRACTOR


def extract_single_feature(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    try:
        ext = get_extractor()
        with Image.open(rec["local_path"]) as im:
            canvas_arr = ArtworkTiler.crop_frame(im.convert("RGB"))
            crop = np.asarray(Image.fromarray(canvas_arr).resize((224, 224)))
        res = ext.extract_patch(crop)
        return res.vector
    except Exception:
        return None


def main() -> None:
    t0 = time.time()
    records = load_all_records()
    logger.info("Total catalogued paintings with verified files: %d", len(records))

    cache_dir = Path("data/embeddings/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "facture_cache_all.npz"

    cache: Dict[str, np.ndarray] = {}
    if cache_path.exists():
        try:
            with np.load(cache_path) as data:
                cache = {k: data[k] for k in data.files}
            logger.info("Loaded %d vectors from persistent facture cache", len(cache))
        except Exception as e:
            logger.warning("Could not load cache: %s", e)

    missing_records = [r for r in records if r["artwork_id"] not in cache]
    if missing_records:
        logger.info("Extracting spatial facture vectors for %d new paintings (workers=12)...", len(missing_records))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=12) as ex:
            new_vectors = list(ex.map(extract_single_feature, missing_records))
        for r, v in zip(missing_records, new_vectors):
            if v is not None and not np.isnan(v).any():
                cache[r["artwork_id"]] = v
        elapsed = time.time() - t0
        logger.info("Extracted %d features in %.1f seconds (%.1f img/s)",
                    len(missing_records), elapsed, len(missing_records) / max(elapsed, 0.1))
        # Save updated cache
        try:
            np.savez_compressed(cache_path, **cache)
            logger.info("Updated persistent facture cache with %d entries", len(cache))
        except Exception as e:
            logger.warning("Could not save cache: %s", e)

    kept_records = []
    kept_vectors = []
    for r in records:
        v = cache.get(r["artwork_id"])
        if v is not None and not np.isnan(v).any():
            kept_records.append(r)
            kept_vectors.append(v)

    elapsed = time.time() - t0
    logger.info("Successfully extracted features for %d paintings in %.1f seconds (%.1f img/s)",
                len(kept_records), elapsed, len(kept_records) / max(elapsed, 0.1))

    X = np.stack(kept_vectors, axis=0)

    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    from sklearn.preprocessing import RobustScaler

    Xs = RobustScaler().fit_transform(X)
    Xs = np.clip(Xs, -8, 8)

    pca = PCA(n_components=min(50, Xs.shape[1], len(Xs) - 1), random_state=42)
    Z = pca.fit_transform(Xs)
    logger.info("PCA %d components retains %.1f%% variance", Z.shape[1], 100 * pca.explained_variance_ratio_.sum())

    import umap
    logger.info("Computing 2D and 3D UMAP manifold projections...")
    reducer2 = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.1, random_state=42)
    reducer3 = umap.UMAP(n_components=3, n_neighbors=25, min_dist=0.1, random_state=42)
    emb2 = reducer2.fit_transform(Z)
    emb3 = reducer3.fit_transform(Z)

    hdb = HDBSCAN(min_cluster_size=65, min_samples=12, cluster_selection_method="leaf", copy=True)
    raw_labels = hdb.fit_predict(emb3)
    unique_clusters = sorted(list(set(raw_labels) - {-1}))
    centroids = np.array([emb3[raw_labels == c].mean(axis=0) for c in unique_clusters])
    dists = np.linalg.norm(emb3[:, None, :] - centroids[None, :, :], axis=2)
    nearest_idx = np.argmin(dists, axis=1)
    min_dists = dists[np.arange(len(emb3)), nearest_idx]
    outlier_thresh = np.percentile(min_dists, 94.5)
    labels = np.array([unique_clusters[i] for i in nearest_idx], dtype=int)
    is_outlier_arr = min_dists > outlier_thresh
    labels[is_outlier_arr] = -1
    probs = np.clip(1.0 - (min_dists / (max(outlier_thresh, 1e-5) * 1.15)), 0.05, 1.0)
    probs[is_outlier_arr] = 0.0
    n_clusters = len(unique_clusters)
    noise_fraction = float(np.mean(labels == -1))
    mask = labels != -1

    museums = [r["museum"] for r in kept_records]
    centuries = [r["century"] for r in kept_records]
    ami_museum = float(adjusted_mutual_info_score(np.array(museums)[mask], labels[mask])) if n_clusters > 1 else 0.0
    ami_century = float(adjusted_mutual_info_score(np.array(centuries)[mask], labels[mask])) if n_clusters > 1 else 0.0
    sil = float(silhouette_score(Z[mask], labels[mask], sample_size=min(3000, int(mask.sum())), random_state=42)) if n_clusters > 1 else float("nan")

    logger.info("Clustering Summary: %d clusters, Noise: %.1f%%, AMI(museum): %.4f, AMI(century): %.4f",
                n_clusters, noise_fraction * 100, ami_museum, ami_century)

    rows = []
    for r, lbl, p, e2, e3 in zip(kept_records, labels, probs, emb2, emb3):
        # Clean relative path for both HTTP and local file view
        rel_p = Path(r["local_path"]).as_posix()
        rows.append({
            "artwork_id": r["artwork_id"],
            "title": r["title"],
            "artist": r["artist"],
            "creation_date": r["creation_date"],
            "century": int(r["century"]),
            "museum": r["museum"],
            "cluster_id": int(lbl),
            "membership_probability": round(float(p), 4),
            "is_outlier": bool(lbl == -1),
            "x2": round(float(e2[0]), 4),
            "y2": round(float(e2[1]), 4),
            "x3": round(float(e3[0]), 4),
            "y3": round(float(e3[1]), 4),
            "z3": round(float(e3[2]), 4),
            "local_path": rel_p,
        })

    df = pd.DataFrame(rows)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out_dir / "collection_clustering.parquet", index=False)
    df.to_csv(out_dir / "collection_clustering.csv", index=False, encoding="utf-8")

    summary = {
        "n_artworks": len(df),
        "n_clusters": int(n_clusters),
        "noise_fraction": round(noise_fraction, 4),
        "silhouette": round(sil, 4) if not np.isnan(sil) else None,
        "ami_cluster_vs_museum": round(ami_museum, 4),
        "ami_cluster_vs_century": round(ami_century, 4),
        "projection": "umap",
    }
    with open(out_dir / "collection_clustering_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export interactive 3D WebGL point-cloud
    export_manifold_explorer(df, out_dir / "collection_manifold_3d.html", summary)
    logger.info("[OK] Exported 3D manifold with %d artworks to data/processed/collection_manifold_3d.html", len(df))


if __name__ == "__main__":
    main()
