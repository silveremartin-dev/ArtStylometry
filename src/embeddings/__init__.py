"""
Dual-Scale Feature Extraction & Embedding Engine.
Micro-texture brushstroke analysis (DINOv2 / Gram matrices) and macro-compositional layout.
"""

from .micro_extractor import MicroFactureExtractor, PatchEmbeddingResult
from .macro_extractor import MacroCompositionExtractor, CompositionFeatures
from .aggregator import PatchAggregator, ArtworkDescriptor

__all__ = [
    "MicroFactureExtractor",
    "PatchEmbeddingResult",
    "MacroCompositionExtractor",
    "CompositionFeatures",
    "PatchAggregator",
    "ArtworkDescriptor",
]
