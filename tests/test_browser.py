"""
Unit tests for the interactive artwork visual browser and connoisseurship catalog.
"""

from pathlib import Path
import pytest

from src.browser.catalog_data import get_curated_artists_catalog, ArtistCatalogEntry
from src.browser.renderer import ArtworkBrowserRenderer


def test_catalog_data_integrity():
    catalog = get_curated_artists_catalog()
    assert len(catalog) >= 6

    # Verify representations of Musée du Louvre and Smithsonian
    museums = [a.primary_museum for a in catalog]
    assert any("Louvre" in m for m in museums)
    assert any("Smithsonian" in m for m in museums)

    # Verify Leonardo da Vinci
    leonardo = next(a for a in catalog if a.id == "leonardo_da_vinci")
    assert leonardo.birth_year == 1452
    assert leonardo.period == "High Renaissance"
    assert len(leonardo.artworks) >= 4

    # Verify famous case study: Salvator Mundi
    salvator = next(art for art in leonardo.artworks if art.id == "leonardo_salvator_mundi")
    assert salvator.is_case_study is True
    assert salvator.attribution_confidence < 0.50
    assert len(salvator.anomaly_zones) >= 1
    assert "Boltraffio" in salvator.case_study_summary

    # Verify famous case study: The Man with the Golden Helmet (Rembrandt de-attribution)
    rembrandt = next(a for a in catalog if a.id == "rembrandt_van_rijn")
    helmet = next(art for art in rembrandt.artworks if art.id == "rembrandt_golden_helmet")
    assert helmet.is_case_study is True
    assert helmet.attribution_confidence < 0.30

    # Verify famous case study: Vermeer's Girl with a Flute (2022 NGA de-attribution)
    vermeer = next(a for a in catalog if a.id == "johannes_vermeer")
    flute = next(art for art in vermeer.artworks if art.id == "vermeer_girl_flute")
    assert flute.is_case_study is True
    assert flute.attribution_confidence < 0.35


def test_browser_html_rendering():
    renderer = ArtworkBrowserRenderer()
    html = renderer.render_html()

    assert "<!DOCTYPE html>" in html
    assert "ArtStylometry" in html
    assert "artistSearchInput" in html
    assert "btnSortChronological" in html
    assert "btnSortAlphabetical" in html
    assert "Leonardo da Vinci" in html
    assert "James Abbott McNeill Whistler" in html
    assert "Mona Lisa" in html
    assert "Salvator Mundi" in html


def test_browser_export(tmp_path):
    renderer = ArtworkBrowserRenderer()
    out_file = tmp_path / "test_browser.html"
    res = renderer.export(out_file)

    assert res.exists()
    assert res.stat().st_size > 5000
    content = res.read_text(encoding="utf-8")
    assert "Museum Artwork Browser" in content
