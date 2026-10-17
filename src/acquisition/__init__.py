"""
IIIF and Museum Archive Data Acquisition Engine.
Interrogates IIIF Presentation and Image APIs to harvest native gigapixel tiles.
"""

from .iiif_client import IIIFClient, IIIFManifestInfo, IIIFImageService
from .downloader import TileDownloader, CanvasReconstructor

__all__ = [
    "IIIFClient",
    "IIIFManifestInfo",
    "IIIFImageService",
    "TileDownloader",
    "CanvasReconstructor",
]
