"""
Scratch Experiment: End-to-End Stylometry Demonstration.
Synthesizes canvas textures mimicking master sfumato, pupil brushwork,
generates spatial heatmaps, and tests unsupervised clustering.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ArtStylometryPipeline


def create_synthetic_canvas(style_type: str, seed: int = 42) -> Image.Image:
    """Creates a synthetic painting canvas with distinct textural characteristics."""
    np.random.seed(seed)
    h, w = 512, 512

    if style_type == "leonardo_sfumato":
        # Smooth radial gradients, dark background, delicate soft transitions
        y, x = np.mgrid[:h, :w]
        dist = np.hypot(x - w // 2, y - h // 2)
        base = np.clip(180 - dist * 0.4, 20, 220)
        noise = np.random.normal(0, 4, (h, w))
        canvas = np.clip(base + noise, 0, 255).astype(np.uint8)
        rgb = np.stack([canvas, (canvas * 0.9).astype(np.uint8), (canvas * 0.75).astype(np.uint8)], axis=-1)

    elif style_type == "pupil_hatching":
        # Sharp high-frequency linear brushwork / hatching
        canvas = np.random.randint(60, 160, (h, w), dtype=np.uint8)
        # Add diagonal brush strokes
        for i in range(0, h + w, 8):
            rr = np.arange(max(0, i - w), min(h, i + 1))
            cc = i - rr
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            canvas[rr[valid], cc[valid]] = np.clip(canvas[rr[valid], cc[valid]] + 40, 0, 255)
        rgb = np.stack([canvas, (canvas * 0.85).astype(np.uint8), (canvas * 0.7).astype(np.uint8)], axis=-1)

    elif style_type == "composite_workshop":
        # Leonardo sfumato in upper center (face), pupil hatching in lower half (drapery)
        face_img = np.array(create_synthetic_canvas("leonardo_sfumato", seed))
        pupil_img = np.array(create_synthetic_canvas("pupil_hatching", seed + 1))
        rgb = face_img.copy()
        # Bottom half and right third painted by pupil
        rgb[256:, :] = pupil_img[256:, :]
        rgb[:, 380:] = pupil_img[:, 380:]

    else:
        # High key bright interior
        rgb = np.random.randint(180, 240, (h, w, 3), dtype=np.uint8)

    return Image.fromarray(rgb)


def main():
    scratch_dir = Path(__file__).resolve().parent
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Generating synthetic artwork corpus...")
    ref_1 = scratch_dir / "leonardo_mona_lisa_sim.jpg"
    ref_2 = scratch_dir / "leonardo_st_john_sim.jpg"
    ref_3 = scratch_dir / "leonardo_virgin_rocks_sim.jpg"
    composite_path = scratch_dir / "salvator_mundi_sim_composite.jpg"
    unrelated_path = scratch_dir / "vermeer_milkmaid_sim.jpg"

    create_synthetic_canvas("leonardo_sfumato", seed=101).save(ref_1)
    create_synthetic_canvas("leonardo_sfumato", seed=102).save(ref_2)
    create_synthetic_canvas("leonardo_sfumato", seed=103).save(ref_3)
    create_synthetic_canvas("composite_workshop", seed=201).save(composite_path)
    create_synthetic_canvas("unrelated_bright", seed=301).save(unrelated_path)

    print("[*] Running ArtStylometry pipeline audit on composite artwork...")
    pipeline = ArtStylometryPipeline(patch_size=128, stride=64)
    profile = pipeline.build_artist_profile("Leonardo da Vinci", [ref_1, ref_2, ref_3])

    report_html = pipeline.audit_artwork(
        artwork_path=composite_path,
        reference_profile=profile,
        output_dir=scratch_dir / "audits",
    )
    print(f"[OK] Audit completed! View interactive report at: {report_html}")

    print("\n[*] Clustering collection...")
    all_artworks = [ref_1, ref_2, ref_3, composite_path, unrelated_path]
    cluster_res, proj, _ = pipeline.cluster_collection(all_artworks, min_cluster_size=2)
    print(f"[OK] Discovered {cluster_res.num_clusters} clusters with {cluster_res.num_outliers} outliers.")
    for aid, lbl in zip(cluster_res.artwork_ids, cluster_res.labels):
        print(f"  [{lbl}] {aid}")


if __name__ == "__main__":
    main()
