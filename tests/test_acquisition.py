"""
Unit tests for IIIF Manifest client and tile reconstruction.
"""

import json
import pytest
from PIL import Image
import numpy as np

from src.acquisition.iiif_client import IIIFClient, IIIFImageService, IIIFTileSpec
from src.acquisition.downloader import CanvasReconstructor, TileDownloader


def test_iiif_image_service_url_construction():
    service = IIIFImageService(
        service_url="https://iiif.micr.io/demo",
        width=4000,
        height=3000,
        tiles=[IIIFTileSpec(width=512, height=512, scale_factors=[1, 2, 4])],
    )
    url = service.construct_region_url(region=(100, 200, 512, 512), size=(256, 256), output_format="jpg")
    assert url == "https://iiif.micr.io/demo/100,200,512,512/256,256/0/default.jpg"


def test_iiif_client_parse_v2_manifest():
    client = IIIFClient()
    v2_data = {
        "@context": "http://iiif.io/api/presentation/2/context.json",
        "@id": "https://example.org/manifest/v2",
        "label": "The Night Watch",
        "metadata": [
            {"label": "Artist", "value": "Rembrandt van Rijn"},
            {"label": "Date", "value": "1642"},
            {"label": "Repository", "value": "Rijksmuseum"},
        ],
        "sequences": [
            {
                "canvases": [
                    {
                        "@id": "https://example.org/canvas/1",
                        "label": "Canvas 1",
                        "width": 8000,
                        "height": 6000,
                        "images": [
                            {
                                "resource": {
                                    "service": {
                                        "@id": "https://example.org/image/rembrandt"
                                    }
                                }
                            }
                        ],
                    }
                ]
            }
        ],
    }
    info = client.parse_manifest(v2_data, "https://example.org/manifest/v2")
    assert info.label == "The Night Watch"
    assert info.artist == "Rembrandt van Rijn"
    assert info.date == "1642"
    assert info.institution == "Rijksmuseum"
    assert len(info.canvases) == 1
    assert info.primary_image_service is not None
    assert info.primary_image_service.service_url == "https://example.org/image/rembrandt"


def test_iiif_client_parse_v3_manifest():
    client = IIIFClient()
    v3_data = {
        "@context": "http://iiif.io/api/presentation/3/context.json",
        "id": "https://example.org/manifest/v3",
        "label": {"en": ["Mona Lisa"]},
        "metadata": [
            {"label": {"en": ["Creator"]}, "value": {"en": ["Leonardo da Vinci"]}},
            {"label": {"en": ["Date"]}, "value": {"en": ["1503"]}},
        ],
        "items": [
            {
                "id": "https://example.org/canvas/mona",
                "label": {"en": ["Front Canvas"]},
                "width": 7700,
                "height": 5300,
                "items": [
                    {
                        "items": [
                            {
                                "body": {
                                    "service": [
                                        {"id": "https://example.org/image/monalisa"}
                                    ]
                                }
                            }
                        ]
                    }
                ],
            }
        ],
    }
    info = client.parse_manifest(v3_data, "https://example.org/manifest/v3")
    assert info.label == "Mona Lisa"
    assert info.artist == "Leonardo da Vinci"
    assert len(info.canvases) == 1
    assert info.primary_image_service is not None
    assert info.primary_image_service.service_url == "https://example.org/image/monalisa"


def test_load_local_canvas(tmp_path):
    img = Image.new("RGB", (300, 200), color=(120, 80, 40))
    test_file = tmp_path / "test_painting.jpg"
    img.save(test_file)

    loaded = CanvasReconstructor.load_local_canvas(test_file)
    assert loaded.size == (300, 200)
    assert loaded.mode == "RGB"
