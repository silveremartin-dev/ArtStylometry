"""
End-to-end integration tests for ArtStylometry pipeline.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.pipeline import ArtStylometryPipeline


@pytest.fixture
def synthetic_artworks(tmp_path):
    """Generates synthetic reference artworks and one test painting."""
    ref_paths = []
    np.random.seed(42)

    # Reference artworks for Artist A (warm dark chiaroscuro)
    for i in range(3):
        arr = np.random.randint(20, 100, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        p = tmp_path / f"ref_artist_a_{i}.jpg"
        img.save(p)
        ref_paths.append(p)

    # Test artwork with composite hand (half dark master, half bright assistant)
    test_arr = np.random.randint(20, 100, (256, 256, 3), dtype=np.uint8)
    test_arr[128:, 128:] = np.random.randint(180, 250, (128, 128, 3), dtype=np.uint8)
    test_img = Image.fromarray(test_arr)
    test_path = tmp_path / "test_candidate.jpg"
    test_img.save(test_path)

    # Unrelated artwork for Artist B
    unrelated_arr = np.random.randint(150, 240, (256, 256, 3), dtype=np.uint8)
    unrelated_img = Image.fromarray(unrelated_arr)
    unrelated_path = tmp_path / "artist_b_01.jpg"
    unrelated_img.save(unrelated_path)

    return ref_paths, test_path, unrelated_path


def test_pipeline_audit_and_clustering(synthetic_artworks, tmp_path):
    ref_paths, test_path, unrelated_path = synthetic_artworks

    pipeline = ArtStylometryPipeline(
        use_deep_model=False,
        patch_size=64,
        stride=64,
        cache_dir=str(tmp_path / "cache"),
    )

    # 1. Build profile
    profile = pipeline.build_artist_profile("Rembrandt", ref_paths)
    assert profile.num_reference_artworks == 3
    assert profile.num_reference_patches > 0

    # 2. Audit candidate artwork
    report_html = pipeline.audit_artwork(
        artwork_path=test_path,
        reference_profile=profile,
        output_dir=tmp_path / "audits",
    )
    assert report_html.exists()
    assert report_html.name == "index.html"
    content = report_html.read_text(encoding="utf-8")
    assert "ArtStylometry Connoisseurship Audit" in content

    # 3. Cluster collection
    all_artworks = ref_paths + [test_path, unrelated_path]
    cluster_res, proj, descs = pipeline.cluster_collection(all_artworks, min_cluster_size=2)

    assert len(descs) == 5
    assert len(cluster_res.labels) == 5
    assert proj.coordinates.shape == (5, 2)
