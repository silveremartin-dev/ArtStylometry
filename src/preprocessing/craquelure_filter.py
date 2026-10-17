"""
Craquelure Detection & Masking Filter.
Detects age-related micro-crack networks via morphological top-hat transforms
to prevent models from confusing canvas degradation with painterly facture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import ndimage


@dataclass
class CraquelureMask:
    """Mask and statistics representing detected crack networks in a patch."""
    binary_mask: np.ndarray  # (H, W) bool, True where crack is present
    crack_density: float  # Fraction of patch covered by cracks [0, 1]
    cleanliness_score: float  # 1.0 - crack_density [0, 1]
    filtered_luminance: np.ndarray  # Luminance with cracks attenuated


class CraquelureDetector:
    """Isolates fine linear craquelures via morphological black top-hat filtering."""

    def __init__(
        self,
        structuring_element_size: int = 5,
        detection_threshold_percentile: float = 96.0,
        min_contrast: float = 0.08,
    ) -> None:
        """
        Args:
            structuring_element_size: Radius of the disk/square used for morphological closing.
            detection_threshold_percentile: Percentile of top-hat response used for crack thresholding.
            min_contrast: Absolute contrast floor to prevent triggering in flat dark areas.
        """
        self.se_size = structuring_element_size
        self.threshold_percentile = detection_threshold_percentile
        self.min_contrast = min_contrast

    def detect_cracks(self, luminance_patch: np.ndarray) -> CraquelureMask:
        """
        Executes black top-hat filter on grayscale luminance image (H, W) in range [0, 1].
        Black Top-Hat: Closing(I) - I.
        Highlights dark thin structures (cracks) that are smaller than the structuring element.
        """
        lum = np.asarray(luminance_patch, dtype=np.float32)
        if lum.ndim == 3:
            lum = 0.2126 * lum[:, :, 0] + 0.7152 * lum[:, :, 1] + 0.0722 * lum[:, :, 2]
        if lum.max() > 1.0:
            lum /= 255.0

        # Create structural footprint
        footprint = ndimage.generate_binary_structure(2, 1)
        footprint = ndimage.iterate_structure(footprint, self.se_size // 2)

        # Morphological closing
        closed = ndimage.grey_closing(lum, footprint=footprint)
        tophat = np.clip(closed - lum, 0.0, 1.0)

        # Dynamic thresholding based on distribution and contrast floor
        thresh_val = max(
            float(np.percentile(tophat, self.threshold_percentile)),
            self.min_contrast,
        )
        binary_mask = tophat >= thresh_val

        # Clean noise: remove isolated tiny specks (area < 3 pixels) without erasing thin linear cracks
        labeled, num_features = ndimage.label(binary_mask)
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            too_small = component_sizes < 3
            too_small_mask = too_small[labeled]
            binary_mask[too_small_mask] = False

        crack_pixels = int(np.sum(binary_mask))
        total_pixels = binary_mask.size
        density = float(crack_pixels / max(1, total_pixels))
        cleanliness = float(max(0.0, 1.0 - density))

        # Attenuate cracks by replacing with closed image values
        filtered = lum.copy()
        filtered[binary_mask] = closed[binary_mask]

        return CraquelureMask(
            binary_mask=binary_mask,
            crack_density=density,
            cleanliness_score=cleanliness,
            filtered_luminance=filtered,
        )
