"""
Interactive Artwork Browser & Connoisseurship Navigator Engine.
"""

from .catalog_data import ArtistCatalogEntry, ArtworkRecord, get_curated_artists_catalog
from .renderer import ArtworkBrowserRenderer

__all__ = [
    "ArtistCatalogEntry",
    "ArtworkRecord",
    "get_curated_artists_catalog",
    "ArtworkBrowserRenderer",
]
