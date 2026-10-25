"""
Spatial Attribution Heatmap Generator for Contested Masters.
Generates 2D continuous confidence heatmaps and anomaly overlay images for
identified outlier works ('moutons noirs', studio assistants, and fakes).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attribution.reference_profile import ArtistReferenceProfile, DistanceMetric
from src.attribution.heatmap_generator import AttributionHeatmapGenerator
from src.visualization.heatmap_overlay import HeatmapOverlayRenderer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_heatmaps_for_master(artist_key: str, max_works: int = 5) -> list[Path]:
    corpus_dir = Path("data/raw/contested_masters") / artist_key
    cat_file = corpus_dir / "catalog.json"
    cache_dir = corpus_dir / ".features_cache"

    if not cat_file.exists() or not cache_dir.exists():
        logger.warning(f"Corpus or features cache missing for {artist_key}")
        return []

    with open(cat_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Autograph references
    autographs = [r for r in records if r["attribution_category"] == "autograph"]
    if len(autographs) < 2:
        autographs = [r for r in records if r["attribution_category"] != "imitator_fake"]

    patch_vecs = []
    for r in autographs:
        npz_file = cache_dir / f"{r['artwork_id']}.npz"
        if npz_file.exists():
            npz = np.load(npz_file)
            patch_vecs.append(npz["patch_vecs"])

    if not patch_vecs:
        return []

    ref_patches = np.vstack(patch_vecs)
    profile = ArtistReferenceProfile.fit_from_patch_vectors(
        artist_name=records[0].get("artist_canonical", artist_key),
        patch_vectors=ref_patches,
        num_artworks=len(autographs),
        covariance_method="ledoit_wolf",
    )

    out_dir = Path("data/processed/heatmaps") / artist_key
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = AttributionHeatmapGenerator(patch_size=256, stride=128)
    renderer = HeatmapOverlayRenderer(colormap_name="inferno", alpha=0.50)

    # Focus on non-autograph or contested works first
    target_records = [r for r in records if r["attribution_category"] in ("workshop", "follower", "imitator_fake")]
    # If none, take all
    if not target_records:
        target_records = records[:max_works]

    generated_paths = []
    for r in target_records[:max_works]:
        img_path = Path(r["local_path"])
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as pil_img:
                # Downsample slightly for heatmap tiling speed if gigapixel
                w, h = pil_img.size
                if max(w, h) > 2000:
                    scale = 2000 / max(w, h)
                    img_for_hm = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
                else:
                    img_for_hm = pil_img.copy()

                hm_res = generator.generate(img_for_hm, profile)
                overlay = renderer.render_overlay(img_for_hm, hm_res, highlight_anomalies=True)

                out_file = out_dir / f"{r['artwork_id']}_heatmap.jpg"
                overlay.save(out_file, quality=90, optimize=True)
                generated_paths.append(out_file)
                logger.info(f"Generated heatmap for {r['title']} -> {out_file} ({len(hm_res.anomaly_zones)} anomaly zones)")
        except Exception as e:
            logger.error(f"Error generating heatmap for {r['artwork_id']}: {e}")

    return generated_paths


def main():
    masters = ["vermeer", "leonardo", "rembrandt", "hals", "cranach", "rubens", "titian", "greco", "raphael", "goya", "velazquez", "bosch"]
    total = 0
    for m in masters:
        paths = generate_heatmaps_for_master(m, max_works=4)
        total += len(paths)
    print(f"\n[OK] Successfully generated {total} 2D spatial heatmap overlays in data/processed/heatmaps/!")


if __name__ == "__main__":
    main()
