"""
Collection-Scale Stylometric Clustering (tens of thousands of paintings).

Pipeline per painting (cached to data/embeddings/cache/<artwork_id>.npz):
  1. Native-resolution crops (no global down-sampling) chosen by luminance variance.
  2. DINOv2 CLS embeddings of the crops (self-supervised, texture-level) -> mean & std pooling.
  3. Analytical facture vector (gradient statistics, Gram matrix, steerable filters,
     ductus coherence / orientation entropy) on 256 px native crops.

Collection-level analysis:
  - Robust standardisation, PCA (50 d) -> HDBSCAN density clustering (noise = outliers).
  - UMAP 2D + 3D projections for the interactive manifold explorer.
  - Confound audit: Adjusted Mutual Information between clusters and (a) holding museum,
    (b) century. High museum-AMI means the embedding is clustering by digitisation pipeline
    rather than by painterly style (AGENTS.md bottleneck #2) and must be reported.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.dino_texture import DinoTextureEncoder, select_native_crops  # noqa: E402
from src.embeddings.micro_extractor import MicroFactureExtractor  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cluster_at_scale")

CACHE_DIR = Path("data/embeddings/cache")


def parse_century(date_str: str) -> Optional[int]:
    m = re.search(r"(1[2-9]\d\d|20\d\d)", str(date_str))
    return int(m.group(1)) // 100 + 1 if m else None


def load_catalogs(patterns: List[str]) -> List[Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for pat in patterns:
        for path in glob.glob(pat):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("Skipping unreadable catalog %s (%s)", path, e)
                continue
            for r in data:
                aid = r.get("artwork_id")
                if aid and aid not in records and Path(r.get("local_path", "")).exists():
                    r.setdefault("artist", r.get("artist_canonical", "Unknown"))
                    records[aid] = r
    return list(records.values())


def extract_features(rec: Dict[str, Any], dino: DinoTextureEncoder, micro: MicroFactureExtractor) -> Optional[np.ndarray]:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", rec["artwork_id"])
    cache = CACHE_DIR / f"{safe}.npz"
    if cache.exists():
        try:
            return np.load(cache)["vec"]
        except Exception:
            pass
    with Image.open(rec["local_path"]) as im:
        img = im.convert("RGB")
    crops224 = select_native_crops(img, crop_size=224, grid=5, keep=8)
    parts: List[np.ndarray] = []
    if dino.available:
        emb = dino.encode(crops224)
        parts += [emb.mean(axis=0), emb.std(axis=0)]
    crops256 = select_native_crops(img, crop_size=256, grid=4, keep=4)
    an = np.array([micro.extract_patch(c, i).vector for i, c in enumerate(crops256)], dtype=np.float32)
    parts.append(an.mean(axis=0))
    vec = np.concatenate(parts).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, vec=vec)
    return vec


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster the full harvested collection")
    ap.add_argument("--catalogs", nargs="+", default=["data/raw/paintings_catalog*.json", "data/raw/contested_masters/*/catalog.json"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--output-dir", default="data/processed")
    args = ap.parse_args()

    records = load_catalogs(args.catalogs)
    if args.limit:
        records = records[: args.limit]
    logger.info("Loaded %d paintings present on disk", len(records))

    dino = DinoTextureEncoder(enabled=not args.no_dino)
    micro = MicroFactureExtractor(use_deep_model=False, embedding_dim=128)

    vecs, kept = [], []
    t0 = time.time()
    for i, rec in enumerate(records):
        try:
            v = extract_features(rec, dino, micro)
        except Exception as e:
            logger.warning("Feature extraction failed for %s: %s", rec["artwork_id"], e)
            continue
        if v is None:
            continue
        vecs.append(v)
        kept.append(rec)
        if (i + 1) % 250 == 0:
            rate = (i + 1) / (time.time() - t0)
            logger.info("  %d/%d paintings (%.1f img/s, ETA %.0f min)", i + 1, len(records), rate, (len(records) - i - 1) / rate / 60)

    dims = {len(v) for v in vecs}
    if len(dims) > 1:  # mixed cache (with/without DINO) -> keep majority dimension
        target = max(dims, key=lambda d: sum(len(v) == d for v in vecs))
        pairs = [(v, r) for v, r in zip(vecs, kept) if len(v) == target]
        vecs, kept = [p[0] for p in pairs], [p[1] for p in pairs]
    X = np.stack(vecs)
    logger.info("Feature matrix %s", X.shape)

    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
    from sklearn.preprocessing import RobustScaler

    Xs = RobustScaler().fit_transform(X)
    Xs = np.clip(Xs, -8, 8)
    pca = PCA(n_components=min(args.pca_dim, Xs.shape[1], len(Xs) - 1), random_state=42)
    Z = pca.fit_transform(Xs)
    logger.info("PCA %d d retains %.1f%% variance", Z.shape[1], 100 * pca.explained_variance_ratio_.sum())

    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=max(5, args.min_cluster_size // 3), cluster_selection_method="leaf")
    labels = hdb.fit_predict(Z)
    probs = hdb.probabilities_
    n_clusters = len(set(labels) - {-1})
    noise = float(np.mean(labels == -1))
    mask = labels != -1
    sil = float(silhouette_score(Z[mask], labels[mask], sample_size=min(5000, int(mask.sum())), random_state=42)) if n_clusters > 1 else float("nan")

    try:
        import umap
        emb2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z)
        emb3 = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1, random_state=42).fit_transform(Z)
        proj = "umap"
    except Exception as e:
        logger.warning("UMAP unavailable (%s) -> PCA projection", e)
        emb2, emb3, proj = Z[:, :2], Z[:, :3], "pca"

    museums = [str(r.get("museum", "Unknown")) for r in kept]
    centuries = [parse_century(r.get("creation_date", "")) or 0 for r in kept]
    ami_museum = float(adjusted_mutual_info_score(np.array(museums)[mask], labels[mask])) if n_clusters > 1 else 0.0
    ami_century = float(adjusted_mutual_info_score(np.array(centuries)[mask], labels[mask])) if n_clusters > 1 else 0.0

    rows = []
    for r, lbl, p, e2, e3, c in zip(kept, labels, probs, emb2, emb3, centuries):
        rows.append({
            "artwork_id": r["artwork_id"], "title": str(r.get("title", "")), "artist": str(r.get("artist", "Unknown")),
            "creation_date": str(r.get("creation_date", "")), "century": int(c), "museum": str(r.get("museum", "")),
            "cluster_id": int(lbl), "membership_probability": round(float(p), 4), "is_outlier": bool(lbl == -1),
            "x2": round(float(e2[0]), 4), "y2": round(float(e2[1]), 4),
            "x3": round(float(e3[0]), 4), "y3": round(float(e3[1]), 4), "z3": round(float(e3[2]), 4),
            "local_path": str(r.get("local_path", "")),
        })
    df = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "collection_clustering.parquet", index=False)
    df.drop(columns=["local_path"]).to_csv(out / "collection_clustering.csv", index=False, encoding="utf-8")

    summary = {
        "n_paintings": int(len(df)), "feature_dim": int(X.shape[1]), "dino": bool(dino.available),
        "pca_dim": int(Z.shape[1]), "pca_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "n_clusters": n_clusters, "noise_fraction": round(noise, 4), "silhouette_pca": round(sil, 4) if sil == sil else None,
        "ami_cluster_vs_museum": round(ami_museum, 4), "ami_cluster_vs_century": round(ami_century, 4),
        "projection": proj, "min_cluster_size": args.min_cluster_size,
        "cluster_sizes": {int(k): int(v) for k, v in pd.Series(labels).value_counts().items()},
    }
    (out / "collection_clustering_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("SUMMARY %s", json.dumps({k: v for k, v in summary.items() if k != "cluster_sizes"}))

    from src.visualization.manifold_explorer import export_manifold_explorer
    html = export_manifold_explorer(df, out / "collection_manifold_3d.html", summary)
    logger.info("3D explorer -> %s", html)


if __name__ == "__main__":
    main()
