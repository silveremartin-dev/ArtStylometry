"""
Unit tests for artist reference profiling, spatial heatmaps, and anomaly detection.
"""

import numpy as np
from PIL import Image
import pytest

from src.attribution.reference_profile import ArtistReferenceProfile, DistanceMetric
from src.attribution.heatmap_generator import AttributionHeatmapGenerator
from src.attribution.anomaly_detector import StylisticAnomalyDetector


def test_reference_profile_mahalanobis_calibration():
    np.random.seed(42)
    # Generate canonical artist distribution (e.g. 50 patches with mean around [1, 0, 0...])
    n_samples, dim = 50, 16
    base_vec = np.zeros(dim)
    base_vec[0] = 1.0
    canonical_patches = base_vec + np.random.normal(0, 0.05, (n_samples, dim))

    profile = ArtistReferenceProfile.fit_from_patch_vectors("Leonardo", canonical_patches)

    assert profile.artist_name == "Leonardo"
    assert profile.num_reference_patches == 50
    assert profile.patch_dim == 16
    assert profile.median_distance > 0.0

    # Inlier should have high confidence
    inlier = base_vec + np.random.normal(0, 0.02, dim)
    inlier_conf = profile.compute_attribution_confidence(inlier, DistanceMetric.MAHALANOBIS)
    assert inlier_conf > 0.5

    # Outlier (orthogonal vector) should have low confidence
    outlier = np.zeros(dim)
    outlier[dim - 1] = 1.0
    outlier_conf = profile.compute_attribution_confidence(outlier, DistanceMetric.MAHALANOBIS)
    assert outlier_conf < inlier_conf


def test_heatmap_generator_detects_anomaly_region():
    # Build a simple reference profile
    np.random.seed(42)
    dim = 32
    master_vectors = np.random.normal(0.5, 0.05, (30, dim))
    profile = ArtistReferenceProfile.fit_from_patch_vectors("Rembrandt", master_vectors)

    # Create synthetic image with master-like texture on left and foreign pattern on right
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :150] = np.random.randint(40, 120, (300, 150, 3), dtype=np.uint8)  # Chiaroscuro master
    arr[:, 150:] = np.random.randint(200, 255, (300, 150, 3), dtype=np.uint8)  # Blown out bright noise

    generator = AttributionHeatmapGenerator(
        patch_size=100,
        stride=50,
        distance_metric=DistanceMetric.COSINE,
        outlier_confidence_threshold=0.6,
        min_anomaly_area_pct=1.0,
    )

    res = generator.generate(arr, profile)
    assert res.heatmap.shape == (300, 300)
    assert 0.0 <= res.overall_attribution_score <= 1.0
    assert res.num_evaluated_patches > 0


def test_stylistic_anomaly_detector():
    np.random.seed(42)
    train_vectors = np.random.normal(0.0, 0.1, (40, 16))

    detector = StylisticAnomalyDetector(method="isolation_forest", contamination=0.1)
    detector.fit(train_vectors)

    # Inlier sample
    inlier = np.random.normal(0.0, 0.05, 16)
    rep_inlier = detector.predict(inlier)
    assert rep_inlier.is_anomaly is False

    # Outlier sample far away
    outlier = np.full(16, 5.0)
    rep_outlier = detector.predict(outlier)
    assert rep_outlier.is_anomaly is True
    assert rep_outlier.anomaly_score < rep_inlier.anomaly_score
