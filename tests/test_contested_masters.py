"""
Unit tests for Contested Masters Harvester and Corpus Auditor.
Verifies attribution categorization, shrinkage reference modeling, and outlier ranking.
"""

import json
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

from src.acquisition.contested_masters_harvester import ContestedMastersHarvester
from src.attribution.artist_corpus_auditor import ArtistCorpusAuditor, CanvasAuditResult


def test_attribution_categorizer():
    harvester = ContestedMastersHarvester()

    assert harvester.categorize_attribution("Johannes Vermeer") == "autograph"
    assert harvester.categorize_attribution("Rembrandt van Rijn") == "autograph"
    assert harvester.categorize_attribution("Leonardo da Vinci") == "autograph"
    assert harvester.categorize_attribution("Studio of Johannes Vermeer") == "workshop"
    assert harvester.categorize_attribution("Rembrandt van Rijn and Workshop") == "workshop"
    assert harvester.categorize_attribution("Rembrandt Workshop (Possibly Carel Fabritius)") == "workshop"
    assert harvester.categorize_attribution("Andrea del Verrocchio and Leonardo da Vinci") == "workshop"
    assert harvester.categorize_attribution("Follower of Frans Hals") == "follower"
    assert harvester.categorize_attribution("Follower of Leonardo da Vinci") == "follower"
    assert harvester.categorize_attribution("Imitator of Johannes Vermeer") == "imitator_fake"
    assert harvester.categorize_attribution("After Lucas Cranach the Elder") == "imitator_fake"
    assert "leonardo" in harvester.ARTIST_CONFIGS


def test_leonardo_curated_holdings_structure():
    from src.acquisition.contested_masters_harvester import LEONARDO_CURATED_HOLDINGS
    assert len(LEONARDO_CURATED_HOLDINGS) >= 14
    titles = [h["title"] for h in LEONARDO_CURATED_HOLDINGS]
    assert any("Mona Lisa" in t for t in titles)
    assert any("Virgin of the Rocks" in t for t in titles)
    assert any("Salvator Mundi" in t for t in titles)
    assert any("La Bella Principessa" in t for t in titles)


def test_artist_corpus_auditor_identifies_moutons_noirs(tmp_path):
    corpus_dir = tmp_path / "rembrandt_test"
    corpus_dir.mkdir(parents=True)
    catalog_file = corpus_dir / "catalog.json"

    # Create 3 synthetic canvases:
    # 2 canonical works with smooth low-frequency textures
    # 1 outlier with noisy high-frequency texture (mouton noir)
    records = []
    np.random.seed(42)

    for i in range(2):
        # Create subtle smooth brushwork gradient
        x = np.linspace(50, 200, 512, dtype=np.uint8)
        img_array = np.tile(x, (512, 1)).astype(np.uint8)
        img_rgb = np.stack([img_array, img_array, img_array], axis=2)
        img_path = corpus_dir / f"canonical_{i}.jpg"
        Image.fromarray(img_rgb).save(img_path)

        records.append({
            "artwork_id": f"rembrandt_can_{i}",
            "artist_canonical": "Rembrandt van Rijn",
            "title": f"Canonical Self-Portrait {i+1}",
            "official_attribution": "Rembrandt van Rijn",
            "attribution_category": "autograph",
            "creation_date": "1659",
            "medium": "Oil on canvas",
            "dimensions": "80 x 60 cm",
            "museum": "National Gallery of Art",
            "image_url": "https://example.org/can.jpg",
            "local_path": str(img_path.resolve()),
            "width": 512,
            "height": 512,
            "filesize_bytes": 50000,
        })

    # Create outlier painting (mouton noir) with intense random noise / checkerboard
    outlier_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    outlier_path = corpus_dir / "outlier_workshop.jpg"
    Image.fromarray(outlier_array).save(outlier_path)

    records.append({
        "artwork_id": "rembrandt_outlier_1",
        "artist_canonical": "Rembrandt van Rijn",
        "title": "A Girl with a Broom (Studio Assistant)",
        "official_attribution": "Rembrandt Workshop (Possibly Carel Fabritius)",
        "attribution_category": "workshop",
        "creation_date": "1651",
        "medium": "Oil on canvas",
        "dimensions": "107 x 91 cm",
        "museum": "National Gallery of Art",
        "image_url": "https://example.org/outlier.jpg",
        "local_path": str(outlier_path.resolve()),
        "width": 512,
        "height": 512,
        "filesize_bytes": 50000,
    })

    with open(catalog_file, "w", encoding="utf-8") as f:
        json.dump(records, f)

    auditor = ArtistCorpusAuditor(patch_size=256, patch_stride=256, use_deep_model=False)
    summary = auditor.audit_corpus(corpus_dir, catalog_file)

    assert summary.total_artworks_evaluated == 3
    assert len(summary.ranked_artworks) == 3

    # Most anomalous artwork should be the noisy workshop piece
    worst_artwork = summary.ranked_artworks[-1]
    assert worst_artwork.artwork_id == "rembrandt_outlier_1"
    assert worst_artwork.is_black_sheep is True
    assert worst_artwork.mahalanobis_distance > summary.ranked_artworks[0].mahalanobis_distance
