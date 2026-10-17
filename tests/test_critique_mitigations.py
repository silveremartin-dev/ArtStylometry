"""
Unit tests for the 5 critique mitigations:
1. Multimodal fusion (Visible, IRR, XRR, UVF)
2. Frequency bandpass filtering (craquelure & varnish decoupling)
3. Ledoit-Wolf & OAS shrinkage covariance (sample size starvation N << D)
4. Metric-scale patch slicing (sensor & DPI calibration)
5. Spatial window apodization (ViT token border discontinuity attenuation)
"""

import numpy as np
from PIL import Image
import pytest

from src.preprocessing.multimodal import MultimodalArtwork, ModalityType
from src.preprocessing.bandpass_filter import FactureBandpassFilter
from src.preprocessing.tiler import ArtworkTiler
from src.attribution.reference_profile import ArtistReferenceProfile, DistanceMetric
from src.embeddings.micro_extractor import MicroFactureExtractor


def test_mitigation_1_multimodal_fusion():
    """Critique 1: Tests visible-IRR-XRR-UVF gradient cross-layer correlation and retouching penalty."""
    mm_art = MultimodalArtwork(artwork_id="canvas_test")

    # Visible surface: diagonal strokes
    vis_arr = np.zeros((100, 100, 3), dtype=np.float32)
    for i in range(100):
        vis_arr[i, i, :] = 1.0
    mm_art.add_layer(ModalityType.VIS_RGB, vis_arr)

    # Coherent IRR underdrawing following the same strokes
    irr_arr = vis_arr.copy()
    mm_art.add_layer(ModalityType.IRR, irr_arr)

    # Clean UVF layer (no retouch absorption)
    uvf_clean = np.ones((100, 100), dtype=np.float32) * 0.9
    mm_art.add_layer(ModalityType.UVF, uvf_clean)

    assert mm_art.has_subsurface_diagnostics() is True
    coherence = mm_art.compute_cross_layer_coherence()

    assert "irr_coherence" in coherence
    assert coherence["irr_coherence"] > 0.8
    assert coherence["retouch_risk_index"] == 0.0
    assert coherence["composite_multimodal_multiplier"] > 0.85


def test_mitigation_2_bandpass_facture_filtering():
    """Critique 2: Tests frequency bandpass isolation of brushwork from cracks and varnish."""
    bpf = FactureBandpassFilter(low_sigma=1.0, high_sigma=6.0)

    # Create synthetic test patch with low-frequency gradient (varnish) and high-frequency noise (cracks)
    y, x = np.mgrid[0:128, 0:128]
    varnish_gradient = (x + y) / 256.0  # Low frequency
    brushstroke = 0.5 * np.sin(2 * np.pi * x / 16.0)  # Mid frequency facture
    craquelure_noise = np.random.normal(0, 0.2, (128, 128))  # High frequency noise

    synthetic_patch = np.clip(varnish_gradient + brushstroke + craquelure_noise, 0.0, 1.0)
    filtered = bpf.filter_luminance(synthetic_patch)

    assert filtered.shape == (128, 128)
    assert 0.0 <= filtered.min() and filtered.max() <= 1.0

    low_e, mid_e, high_e = bpf.extract_spectral_energy(synthetic_patch)
    assert low_e + mid_e + high_e == pytest.approx(1.0, abs=1e-4)
    assert mid_e > 0.0


def test_mitigation_3_shrinkage_covariance_small_sample():
    """Critique 3: Tests Ledoit-Wolf shrinkage on small sample size N << D."""
    np.random.seed(42)
    # N = 10, D = 32 (extreme sample starvation where empirical covariance is non-invertible)
    n_samples, dim = 10, 32
    vectors = np.random.normal(0.0, 1.0, (n_samples, dim))

    # Fit profile with Ledoit-Wolf shrinkage
    profile = ArtistReferenceProfile.fit_from_patch_vectors(
        "Leonardo",
        vectors,
        covariance_method="ledoit_wolf",
        spatial_decorrelation_stride=1,
    )

    assert profile.shrinkage >= 0.0
    assert profile.condition_number < 1e5
    assert not np.isnan(profile.inv_covariance).any()

    # Verify Mahalanobis distance can be computed without numerical explosion
    test_vec = np.random.normal(0.0, 1.0, dim)
    dist = profile.compute_distance(test_vec, DistanceMetric.MAHALANOBIS)
    assert np.isfinite(dist)
    assert dist > 0.0


def test_mitigation_4_metric_scale_tiler():
    """Critique 4: Tests physical metric patch slicing (standardizing mm rather than pixels)."""
    tiler = ArtworkTiler(patch_size=256, stride=128)

    # 1000x1000 canvas at 10 pixels per mm (100 mm x 100 mm canvas)
    canvas = np.random.randint(50, 200, (1000, 1000, 3), dtype=np.uint8)

    # Slice 20 mm x 20 mm physical patches, resampled to standard 256x256 pixels
    grid = tiler.slice_metric_patches(
        image=canvas,
        pixels_per_mm=10.0,
        target_patch_mm=20.0,  # 20 mm = 200 raw pixels
        output_pixel_size=256,
    )

    assert len(grid) > 0
    first_patch = grid.patches[0]
    first_meta = grid.metadata[0]

    assert first_patch.shape == (256, 256, 3)
    assert first_meta.pixels_per_mm == 10.0
    assert first_meta.physical_mm_size == (20.0, 20.0)


def test_mitigation_5_spatial_apodization():
    """Critique 5: Tests spatial window apodization to smooth patch borders."""
    patch = np.ones((64, 64, 3), dtype=np.uint8) * 200
    # Add a hard border step
    patch[0, :, :] = 50
    patch[:, 0, :] = 50

    tapered = MicroFactureExtractor.apply_spatial_apodization(patch, window_type="hann")
    assert tapered.shape == patch.shape

    # Interior center should remain strong
    center_val = tapered[32, 32, 0]
    assert center_val == pytest.approx(200, abs=10)
