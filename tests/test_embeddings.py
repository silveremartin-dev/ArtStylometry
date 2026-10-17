"""
Unit tests for micro/macro feature extraction and patch aggregation.
"""

import numpy as np
from PIL import Image
import pytest

from src.embeddings.micro_extractor import MicroFactureExtractor
from src.embeddings.macro_extractor import MacroCompositionExtractor
from src.embeddings.aggregator import PatchAggregator


def test_micro_facture_extractor():
    extractor = MicroFactureExtractor(use_deep_model=False, embedding_dim=128)

    np.random.seed(42)
    patch = np.random.randint(50, 220, (128, 128, 3), dtype=np.uint8)

    res = extractor.extract_patch(patch, patch_id=1)
    assert res.patch_id == 1
    assert len(res.vector) == 128
    assert np.isclose(np.linalg.norm(res.vector), 1.0, atol=1e-4)
    assert 0.0 <= res.cleanliness <= 1.0
    assert res.gradient_energy >= 0.0


def test_macro_composition_extractor():
    extractor = MacroCompositionExtractor(target_resolution=(128, 128))

    # Canvas with bright upper-left and dark lower-right
    img_arr = np.zeros((200, 200, 3), dtype=np.uint8)
    img_arr[:100, :100] = 255
    img = Image.fromarray(img_arr)

    feats = extractor.extract(img)
    assert len(feats.feature_vector) == 32
    assert np.isclose(np.linalg.norm(feats.feature_vector), 1.0, atol=1e-4)
    # Upper-left centroid should be < 0.5 in both axes
    cx, cy = feats.luminous_barycenter
    assert cx < 0.5
    assert cy < 0.5
    assert feats.chiaroscuro_index > 0.0


def test_patch_aggregator_statistical_and_vlad():
    extractor = MicroFactureExtractor(use_deep_model=False, embedding_dim=64)

    np.random.seed(42)
    patches = [np.random.randint(40, 230, (64, 64, 3), dtype=np.uint8) for _ in range(10)]
    results = [extractor.extract_patch(p, i) for i, p in enumerate(patches)]

    # Statistical aggregation
    agg_stat = PatchAggregator(method="statistical", include_macro=False)
    desc_stat = agg_stat.aggregate(results, artwork_id="test_art_01")

    assert desc_stat.artwork_id == "test_art_01"
    assert desc_stat.num_valid_patches == 10
    assert np.isclose(np.linalg.norm(desc_stat.feature_vector), 1.0, atol=1e-4)

    # VLAD aggregation
    vectors = np.stack([r.vector for r in results], axis=0)
    agg_vlad = PatchAggregator(method="vlad", num_vlad_centers=4, include_macro=False)
    agg_vlad.fit_vlad_codebook(vectors)
    desc_vlad = agg_vlad.aggregate(results, artwork_id="test_art_vlad")

    assert desc_vlad.artwork_id == "test_art_vlad"
    assert np.isclose(np.linalg.norm(desc_vlad.feature_vector), 1.0, atol=1e-4)
