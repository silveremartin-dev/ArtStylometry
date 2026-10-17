"""
Attribution & Anomaly Detection Engine.
Statistical reference profiling, spatial intra-canvas attribution heatmaps,
and one-class outlier detection.
"""

from .reference_profile import ArtistReferenceProfile, DistanceMetric
from .heatmap_generator import AttributionHeatmapGenerator, CanvasHeatmapResult
from .anomaly_detector import StylisticAnomalyDetector, AnomalyReport

__all__ = [
    "ArtistReferenceProfile",
    "DistanceMetric",
    "AttributionHeatmapGenerator",
    "CanvasHeatmapResult",
    "StylisticAnomalyDetector",
    "AnomalyReport",
]
