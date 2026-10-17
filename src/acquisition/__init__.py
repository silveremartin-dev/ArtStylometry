"""
IIIF and Museum Archive Data Acquisition Engine.
Interrogates IIIF Presentation and Image APIs to harvest native gigapixel tiles.
"""

from .iiif_client import IIIFClient, IIIFManifestInfo, IIIFImageService
from .downloader import TileDownloader, CanvasReconstructor
from .museum_registry import (
    MUSEUM_REGISTRY,
    MuseumPreset,
    list_supported_museums,
    get_museum,
    resolve_artwork_endpoint,
    extract_physical_scale_from_manifest,
)

__all__ = [
    "IIIFClient",
    "IIIFManifestInfo",
    "IIIFImageService",
    "TileDownloader",
    "CanvasReconstructor",
    "MUSEUM_REGISTRY",
    "MuseumPreset",
    "list_supported_museums",
    "get_museum",
    "resolve_artwork_endpoint",
    "extract_physical_scale_from_manifest",
]
