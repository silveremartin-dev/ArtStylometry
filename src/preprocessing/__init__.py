"""
Preprocessing & Noise Neutralization Engine.
Deals with patch slicing, chromatic decoupling, and craquelure filtering.
"""

from .tiler import ArtworkTiler, PatchMetadata, PatchGrid
from .color_decoupling import ChromaticDecoupler, SpatialGradientFeatures
from .craquelure_filter import CraquelureDetector, CraquelureMask

__all__ = [
    "ArtworkTiler",
    "PatchMetadata",
    "PatchGrid",
    "ChromaticDecoupler",
    "SpatialGradientFeatures",
    "CraquelureDetector",
    "CraquelureMask",
]
