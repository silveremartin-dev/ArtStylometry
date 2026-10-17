"""
Unit tests for the Worldwide Museum Registry and IIIF Endpoint Resolvers.
Verifies connector presets, manifest generation, and physical dimension parsing.
"""

import pytest
from src.acquisition.museum_registry import (
    MUSEUM_REGISTRY,
    MuseumPreset,
    list_supported_museums,
    get_museum,
    resolve_artwork_endpoint,
    extract_physical_scale_from_manifest,
)


def test_museum_registry_coverage():
    museums = list_supported_museums()
    assert len(museums) >= 12

    expected_codes = {
        "rijksmuseum",
        "met",
        "nga",
        "national_gallery",
        "louvre",
        "smithsonian",
        "artic",
        "gallica",
        "prado",
        "uffizi",
        "getty",
        "cleveland",
    }
    present_codes = {m["code"] for m in museums}
    for code in expected_codes:
        assert code in present_codes, f"Museum code {code} missing from registry"


def test_museum_endpoint_resolution():
    louvre_res = resolve_artwork_endpoint("louvre", "cl010062370")
    assert louvre_res["code"] == "louvre"
    assert "cl010062370" in louvre_res["manifest_url"]

    met_res = resolve_artwork_endpoint("met", "436535")
    assert met_res["code"] == "met"
    assert "436535" in met_res["manifest_url"]

    with pytest.raises(ValueError):
        resolve_artwork_endpoint("non_existent_museum", "12345")


def test_extract_physical_scale_from_manifest():
    # Synthetic IIIF manifest with physical dimensions service
    manifest = {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@id": "https://data.rijksmuseum.nl/object/SK-A-399/manifest",
        "service": {
            "@context": "http://iiif.io/api/annex/services/physdim/1/context.json",
            "profile": "http://iiif.io/api/annex/services/physdim",
            "physicalScale": 0.05,  # 0.05 mm per pixel
            "physicalUnits": "mm",
        },
    }

    scale_info = extract_physical_scale_from_manifest(manifest)
    assert scale_info is not None
    assert scale_info["physical_scale"] == 0.05
    assert scale_info["physical_units"] == "mm"
    assert scale_info["mm_per_pixel"] == 0.05
    assert scale_info["pixels_per_mm"] == pytest.approx(20.0)
