"""
Intra-Canvas Spatial Attribution & Multi-Hand Heatmap Generator.
Generates localized confidence heatmaps across a painting to reveal
studio assistant interventions, pentimenti, or late restoration overpaints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from scipy import ndimage

from .reference_profile import ArtistReferenceProfile, DistanceMetric
from ..preprocessing.tiler import ArtworkTiler, PatchGrid
from ..embeddings.micro_extractor import MicroFactureExtractor, PatchEmbeddingResult

logger = logging.getLogger(__name__)


@dataclass
class AnomalyZone:
    """Identified spatial region on the canvas exhibiting divergent stylistic facture."""
    zone_id: int
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    mean_confidence: float
    area_pixels: int
    area_percentage: float
    description: str


@dataclass
class CanvasHeatmapResult:
    """Output containing 2D spatial confidence matrix, patch scores, and anomaly zones."""
    artist_name: str
    canvas_dimensions: Tuple[int, int]  # (width, height)
    heatmap: np.ndarray  # 2D float array in [0.0, 1.0] representing local confidence
    overall_attribution_score: float  # Global median confidence
    anomaly_zones: List[AnomalyZone]
    num_evaluated_patches: int
    patch_scores: List[float]


class AttributionHeatmapGenerator:
    """Generates continuous spatial attribution maps across high-resolution artworks."""

    def __init__(
        self,
        micro_extractor: Optional[MicroFactureExtractor] = None,
        patch_size: int = 256,
        stride: int = 128,
        distance_metric: DistanceMetric = DistanceMetric.MAHALANOBIS,
        outlier_confidence_threshold: float = 0.35,
        min_anomaly_area_pct: float = 1.5,
    ) -> None:
        self.extractor = micro_extractor or MicroFactureExtractor(use_deep_model=False)
        self.patch_size = patch_size
        self.stride = stride
        self.metric = distance_metric
        self.outlier_threshold = outlier_confidence_threshold
        self.min_anomaly_pct = min_anomaly_area_pct
        self.tiler = ArtworkTiler(patch_size=patch_size, stride=stride, min_variance=10.0)

    def generate(
        self,
        image: Image.Image | np.ndarray,
        reference_profile: ArtistReferenceProfile,
    ) -> CanvasHeatmapResult:
        """
        Processes an artwork and projects attribution confidence across its 2D surface.
        """
        if self.extractor.embedding_dim != reference_profile.patch_dim:
            self.extractor.embedding_dim = reference_profile.patch_dim
        if isinstance(image, Image.Image):
            canvas_arr = np.array(image.convert("RGB"))
        else:
            canvas_arr = np.asarray(image)

        ch, cw = canvas_arr.shape[:2]
        logger.info(
            "Generating attribution heatmap for %s on canvas (%dx%d)...",
            reference_profile.artist_name,
            cw,
            ch,
        )

        grid: PatchGrid = self.tiler.slice_image(canvas_arr)
        if len(grid) == 0:
            logger.warning("No valid patches extracted from canvas. Returning neutral heatmap.")
            neutral = np.full((ch, cw), 0.5, dtype=np.float32)
            return CanvasHeatmapResult(
                artist_name=reference_profile.artist_name,
                canvas_dimensions=(cw, ch),
                heatmap=neutral,
                overall_attribution_score=0.5,
                anomaly_zones=[],
                num_evaluated_patches=0,
                patch_scores=[],
            )

        # Accumulator matrices for overlapping patches
        score_accumulator = np.zeros((ch, cw), dtype=np.float64)
        weight_accumulator = np.zeros((ch, cw), dtype=np.float64)
        patch_scores: List[float] = []

        # Gaussian blending window for smooth patch transitions
        y_w, x_w = np.ogrid[:self.patch_size, :self.patch_size]
        center = (self.patch_size - 1) / 2.0
        dist_sq = (x_w - center) ** 2 + (y_w - center) ** 2
        sigma = self.patch_size / 3.0
        gaussian_weight = np.exp(-dist_sq / (2 * (sigma ** 2)))

        for patch, meta in grid:
            res = self.extractor.extract_patch(patch, meta.patch_id)
            conf = reference_profile.compute_attribution_confidence(res.vector, self.metric)
            # Modulate by crack cleanliness
            effective_conf = conf * max(0.2, res.cleanliness)
            patch_scores.append(float(effective_conf))

            px, py, pw, ph = meta.x, meta.y, meta.width, meta.height
            gw_sub = gaussian_weight[:ph, :pw]

            score_accumulator[py : py + ph, px : px + pw] += effective_conf * gw_sub
            weight_accumulator[py : py + ph, px : px + pw] += gw_sub

        # Normalize accumulated scores
        valid_mask = weight_accumulator > 0
        heatmap = np.full((ch, cw), 0.5, dtype=np.float32)
        heatmap[valid_mask] = (score_accumulator[valid_mask] / weight_accumulator[valid_mask]).astype(np.float32)

        # Smooth overall heatmap slightly
        heatmap = ndimage.gaussian_filter(heatmap, sigma=max(2.0, self.stride / 8.0))
        heatmap = np.clip(heatmap, 0.0, 1.0)

        overall_score = float(np.median(patch_scores)) if patch_scores else 0.5

        # Detect contiguous anomaly zones (low confidence regions)
        anomaly_zones = self._detect_anomaly_zones(heatmap, cw, ch)

        return CanvasHeatmapResult(
            artist_name=reference_profile.artist_name,
            canvas_dimensions=(cw, ch),
            heatmap=heatmap,
            overall_attribution_score=overall_score,
            anomaly_zones=anomaly_zones,
            num_evaluated_patches=len(grid),
            patch_scores=patch_scores,
        )

    def _detect_anomaly_zones(
        self,
        heatmap: np.ndarray,
        canvas_width: int,
        canvas_height: int,
    ) -> List[AnomalyZone]:
        """Identifies connected components where confidence is below the outlier threshold."""
        low_conf_mask = heatmap < self.outlier_threshold
        # Morphological opening to eliminate tiny isolated dots
        low_conf_mask = ndimage.binary_opening(low_conf_mask, structure=np.ones((5, 5)))

        labeled_mask, num_features = ndimage.label(low_conf_mask)
        zones: List[AnomalyZone] = []
        total_pixels = canvas_width * canvas_height

        for feat_id in range(1, num_features + 1):
            zone_pixels = np.where(labeled_mask == feat_id)
            area = len(zone_pixels[0])
            area_pct = (area / max(1, total_pixels)) * 100.0

            if area_pct >= self.min_anomaly_pct:
                min_y, max_y = int(np.min(zone_pixels[0])), int(np.max(zone_pixels[0]))
                min_x, max_x = int(np.min(zone_pixels[1])), int(np.max(zone_pixels[1]))
                zw = max_x - min_x + 1
                zh = max_y - min_y + 1

                mean_val = float(np.mean(heatmap[zone_pixels]))
                desc = (
                    f"Possible studio assistant hand or restoration overpaint "
                    f"(confidence: {mean_val:.2f}, covers {area_pct:.1f}% of canvas)"
                )

                zones.append(
                    AnomalyZone(
                        zone_id=len(zones) + 1,
                        bounding_box=(min_x, min_y, zw, zh),
                        mean_confidence=mean_val,
                        area_pixels=area,
                        area_percentage=float(area_pct),
                        description=desc,
                    )
                )

        return zones
