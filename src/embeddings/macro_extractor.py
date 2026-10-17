"""
Macro-Compositional Layout Extractor.
Analyzes full-canvas geometric organization, luminous barycenters,
chiaroscuro dynamics, and structural proportions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image

from ..preprocessing.color_decoupling import ChromaticDecoupler


@dataclass
class CompositionFeatures:
    """Extracted macro-compositional parameters for an entire artwork."""
    luminous_barycenter: Tuple[float, float]  # (cx, cy) normalized [0, 1]
    golden_ratio_distance: float  # Euclidean distance from nearest golden section
    symmetry_horizontal: float  # Symmetry score [0, 1]
    symmetry_vertical: float
    chiaroscuro_index: float  # Ratio of extreme shadows to midtones
    aspect_ratio: float  # Width / Height
    feature_vector: np.ndarray  # 32-dim normalized vector


class MacroCompositionExtractor:
    """Extracts high-level compositional geometry and light distribution across the full canvas."""

    def __init__(self, target_resolution: Tuple[int, int] = (512, 512)) -> None:
        self.target_resolution = target_resolution
        self.decoupler = ChromaticDecoupler()

    def extract(self, image: Image.Image | np.ndarray) -> CompositionFeatures:
        """
        Extracts composition features from an artwork image.
        Downsamples to target_resolution for scale-invariant structural geometry.
        """
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = image

        orig_w, orig_h = pil_img.size
        aspect = float(orig_w / max(1, orig_h))

        resized = pil_img.resize(self.target_resolution, Image.Resampling.BILINEAR)
        arr = np.array(resized)

        lum = self.decoupler.rgb_to_luminance(arr)
        h, w = lum.shape

        # 1. Luminous barycenter
        total_lum = np.sum(lum)
        if total_lum > 0:
            y_indices, x_indices = np.indices((h, w))
            cx = float(np.sum(x_indices * lum) / (total_lum * w))
            cy = float(np.sum(y_indices * lum) / (total_lum * h))
        else:
            cx, cy = 0.5, 0.5

        # 2. Golden ratio distance
        # Standard golden section coordinates: ~0.382 and ~0.618
        golden_points = [(0.382, 0.382), (0.382, 0.618), (0.618, 0.382), (0.618, 0.618)]
        min_golden_dist = min(
            np.hypot(cx - gx, cy - gy) for gx, gy in golden_points
        )

        # 3. Symmetry metrics
        lum_flipped_h = np.fliplr(lum)
        diff_h = np.mean(np.abs(lum - lum_flipped_h))
        sym_h = float(max(0.0, 1.0 - diff_h * 2.0))

        lum_flipped_v = np.flipud(lum)
        diff_v = np.mean(np.abs(lum - lum_flipped_v))
        sym_v = float(max(0.0, 1.0 - diff_v * 2.0))

        # 4. Chiaroscuro index (high contrast between deep shadows and highlights)
        shadows = np.sum(lum < 0.15)
        highlights = np.sum(lum > 0.85)
        midtones = np.sum((lum >= 0.15) & (lum <= 0.85))
        chiaroscuro = float((shadows + highlights) / max(1, midtones))

        # 5. Luminance histogram profile (16 bins)
        hist, _ = np.histogram(lum, bins=16, range=(0.0, 1.0))
        hist_norm = hist.astype(np.float32) / max(1, lum.size)

        # 6. Spatial 3x3 quadrant grid luminance
        grid_3x3 = []
        qh, qw = h // 3, w // 3
        for qi in range(3):
            for qj in range(3):
                quad = lum[qi * qh : (qi + 1) * qh, qj * qw : (qj + 1) * qw]
                grid_3x3.append(float(np.mean(quad)))
        grid_arr = np.array(grid_3x3, dtype=np.float32)

        # Assemble compact 32-dim feature vector
        scalar_feats = np.array([
            cx,
            cy,
            float(min_golden_dist),
            sym_h,
            sym_v,
            chiaroscuro,
            aspect,
        ], dtype=np.float32)

        # 16 (hist) + 9 (grid 3x3) + 7 (scalar) = 32 dimensions
        vec = np.concatenate([hist_norm, grid_arr, scalar_feats])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return CompositionFeatures(
            luminous_barycenter=(cx, cy),
            golden_ratio_distance=float(min_golden_dist),
            symmetry_horizontal=sym_h,
            symmetry_vertical=sym_v,
            chiaroscuro_index=chiaroscuro,
            aspect_ratio=aspect,
            feature_vector=vec.astype(np.float32),
        )
