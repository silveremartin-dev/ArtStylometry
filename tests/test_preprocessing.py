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
