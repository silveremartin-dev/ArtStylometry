"""
Unit tests for the BatchHarvester component.
Verifies candidate parsing, resizing logic, and metadata catalog serialization.
"""

import io
import json
from pathlib import Path
import pytest
from PIL import Image
import numpy as np

from src.acquisition.batch_harvester import BatchHarvester, HarvestedPaintingRecord


def test_batch_harvester_initialization(tmp_path):
    dest_dir = tmp_path / "paintings"
    catalog_path = tmp_path / "catalog.json"

    harvester = BatchHarvester(
        dest_dir=dest_dir,
        catalog_path=catalog_path,
        max_dimension=3000,
        concurrency=2,
    )

    assert harvester.dest_dir == dest_dir
    assert harvester.catalog_path == catalog_path
    assert harvester.dest_dir.exists()
    assert len(harvester._catalog) == 0


def test_batch_harvester_catalog_persistence(tmp_path):
    dest_dir = tmp_path / "paintings"
    catalog_path = tmp_path / "catalog.json"

    harvester = BatchHarvester(dest_dir=dest_dir, catalog_path=catalog_path)

    dummy_rec = HarvestedPaintingRecord(
        artwork_id="cma_12345",
        title="Test Painting",
        artist="Test Artist",
        creation_date="1650",
        medium="Oil on canvas",
        dimensions="100 x 80 cm",
        museum="Cleveland Museum of Art",
        source_url="https://example.org/12345",
        image_url="https://example.org/12345.jpg",
        local_path=str(dest_dir / "cma_12345.jpg"),
        width=3000,
        height=2000,
        filesize_bytes=1048576,
    )

    harvester._catalog[dummy_rec.artwork_id] = dummy_rec
    harvester._save_catalog()

    assert catalog_path.exists()
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["artwork_id"] == "cma_12345"

    # Reload from disk
    harvester2 = BatchHarvester(dest_dir=dest_dir, catalog_path=catalog_path)
    assert "cma_12345" in harvester2._catalog
    assert harvester2._catalog["cma_12345"].title == "Test Painting"


def test_batch_harvester_rescale_logic(tmp_path, monkeypatch):
    dest_dir = tmp_path / "paintings"
    catalog_path = tmp_path / "catalog.json"

    harvester = BatchHarvester(
        dest_dir=dest_dir,
        catalog_path=catalog_path,
        max_dimension=1000,
    )

    # Create a synthetic 2000x1500 JPEG image in memory
    synth_img = Image.fromarray(np.random.randint(0, 255, (1500, 2000, 3), dtype=np.uint8))
    img_bytes_io = io.BytesIO()
    synth_img.save(img_bytes_io, format="JPEG", quality=90)
    raw_bytes = img_bytes_io.getvalue()

    # Mock urllib.request.urlopen to return synthetic image
    class MockResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_urlopen(req, timeout=30):
        return MockResponse(raw_bytes)

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    candidate = {
        "artwork_id": "cma_synth_1",
        "title": "Synthetic Masterpiece",
        "artist": "Algorithmic Painter",
        "creation_date": "2026",
        "medium": "Digital",
        "dimensions": "Virtual",
        "museum": "Simulated Museum",
        "source_url": "https://example.org/synth",
        "image_url": "https://example.org/synth.jpg",
    }

    rec = harvester.download_painting(candidate)
    assert rec is not None
    assert rec.artwork_id == "cma_synth_1"
    # Max dimension was 1000, so width 2000 must be downscaled to 1000, height to 750
    assert rec.width == 1000
    assert rec.height == 750
    assert Path(rec.local_path).exists()
