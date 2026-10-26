"""
Unit tests for preprocessing, tiler, chromatic decoupling, and craquelure filters.
"""

import numpy as np
from PIL import Image
import pytest

from src.preprocessing.tiler import ArtworkTiler
from src.preprocessing.color_decoupling import ChromaticDecoupler
from src.preprocessing.craquelure_filter import CraquelureDetector


def test_artwork_tiler_slice_dimensions():
    # Create synthetic textured image 600x400
    np.random.seed(42)
    noise = np.random.randint(50, 200, (400, 600, 3), dtype=np.uint8)
    img = Image.fromarray(noise)

    tiler = ArtworkTiler(patch_size=200, stride=200, min_variance=5.0)
    grid = tiler.slice_image(img)

    assert len(grid) == 6  # 3 across x 2 down = 6 patches
    assert grid.canvas_size == (600, 400)

    for patch, meta in grid:
        assert patch.shape == (200, 200, 3)
        assert meta.width == 200
        assert meta.height == 200
        assert meta.canvas_width == 600
        assert meta.canvas_height == 400
        assert meta.variance > 5.0


def test_tiler_filters_blank_margins():
    # Image with top half black and bottom half textured
    arr = np.zeros((400, 400, 3), dtype=np.uint8)
    np.random.seed(42)
    arr[200:, :] = np.random.randint(50, 250, (200, 400, 3), dtype=np.uint8)

    tiler = ArtworkTiler(patch_size=200, stride=200, min_variance=10.0)
    grid = tiler.slice_image(arr)

    # 4 potential patches total, 2 should be rejected as zero variance
    assert len(grid) == 2
    for _, meta in grid:
        assert meta.y >= 200


def test_chromatic_decoupler_luminance_and_gradients():
    decoupler = ChromaticDecoupler(num_orientation_bins=16)

    # Generate image with horizontal gradient (100, 100, 3)
    x = np.broadcast_to(np.arange(100, dtype=np.float32), (100, 100))
    grad_img = np.stack([x * 2.5, x * 2.5, x * 2.5], axis=-1).astype(np.uint8)

    lum = decoupler.rgb_to_luminance(grad_img)
    assert lum.shape == (100, 100)
    assert 0.0 <= lum.min() and lum.max() <= 1.0

    feats = decoupler.compute_gradients(grad_img)
    assert feats.mean_magnitude > 0.0
    assert len(feats.orientation_histogram) == 16
    assert np.isclose(np.sum(feats.orientation_histogram), 1.0)

    vec = decoupler.extract_gradient_feature_vector(grad_img)
    assert len(vec) == 32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)


def test_craquelure_detector_synthetic_crack():
    detector = CraquelureDetector(structuring_element_size=5)

    # Clean smooth background with a dark synthetic crack line
    clean_surface = np.full((100, 100), 0.8, dtype=np.float32)

    # Clean surface test
    clean_mask = detector.detect_cracks(clean_surface)
    assert clean_mask.crack_density < 0.05
    assert clean_mask.cleanliness_score > 0.95

    # Introduce synthetic crack (dark linear fissure)
    cracked_surface = clean_surface.copy()
    cracked_surface[40:42, 10:90] = 0.1  # Dark line

    cracked_mask = detector.detect_cracks(cracked_surface)
    assert cracked_mask.crack_density > 0.01
    assert cracked_mask.cleanliness_score < clean_mask.cleanliness_score
    assert np.any(cracked_mask.binary_mask[40:42, 10:90])


def test_detect_canvas_bbox_and_frame_isolation():
    np.random.seed(123)
    # Create a 500x500 image with a 40px white mount + 30px gilded frame around a 360x360 dark painting
    framed = np.full((500, 500, 3), 245, dtype=np.uint8)
    # Gilded frame region [40:460, 40:460]
    framed[40:460, 40:460] = np.array([195, 155, 55], dtype=np.uint8)
    # Sharp inner rabbet line at [68:70] and [430:432]
    framed[68:432, 68:432] = 15
    # Textured interior canvas [70:430, 70:430]
    framed[70:430, 70:430] = np.random.randint(35, 120, (360, 360, 3), dtype=np.uint8)

    x0, y0, x1, y1 = ArtworkTiler.detect_canvas_bbox(framed)
    assert 60 <= x0 <= 80
    assert 60 <= y0 <= 80
    assert 420 <= x1 <= 440
    assert 420 <= y1 <= 440

    cropped = ArtworkTiler.crop_frame(framed)
    assert cropped.shape[0] < 380 and cropped.shape[1] < 380


def test_algorithmic_artwork_deduplicator(tmp_path):
    from src.preprocessing.deduplicator import AlgorithmicArtworkDeduplicator

    # Create two images of the same painting: one high-res (400x400) and one low-res (200x200) with slight noise
    np.random.seed(42)
    base_art = np.random.randint(40, 220, (200, 200, 3), dtype=np.uint8)
    img_low = Image.fromarray(base_art)
    img_high = img_low.resize((400, 400), Image.Resampling.LANCZOS)
    # Third completely different painting
    diff_art = np.random.randint(40, 220, (400, 400, 3), dtype=np.uint8)
    img_diff = Image.fromarray(diff_art)

    p_low = tmp_path / "low.jpg"
    p_high = tmp_path / "high.jpg"
    p_diff = tmp_path / "diff.jpg"
    img_low.save(p_low)
    img_high.save(p_high)
    img_diff.save(p_diff)

    records = [
        {"artwork_id": "art_low", "local_path": str(p_low), "title": "Masterpiece (Low Res)"},
        {"artwork_id": "art_high", "local_path": str(p_high), "title": "Masterpiece (High Res)"},
        {"artwork_id": "art_diff", "local_path": str(p_diff), "title": "Different Painting"},
    ]

    dedup = AlgorithmicArtworkDeduplicator()
    kept, pruned = dedup.deduplicate_records(records)

    # Must keep high-res and diff, and prune low-res
    assert len(kept) == 2
    assert len(pruned) == 1
    kept_ids = {r["artwork_id"] for r in kept}
    assert "art_high" in kept_ids
    assert "art_diff" in kept_ids
    assert pruned[0]["duplicate_id"] == "art_low"
    assert pruned[0]["retained_id"] == "art_high"
