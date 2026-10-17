"""
Visualization & Reporting Engine.
Produces blended attribution overlays, patch comparisons, and interactive HTML dashboards.
"""

from .heatmap_overlay import HeatmapOverlayRenderer
from .dashboard import StylometryDashboardGenerator

__all__ = [
    "HeatmapOverlayRenderer",
    "StylometryDashboardGenerator",
]
