"""
Chromatic Decoupling & Spatial Gradient Engine.
Strips scanner color bias and extracts pure painterly facture, brushstroke orientation,
and luminance transitions (disegno over colore).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy import ndimage


@dataclass
class SpatialGradientFeatures:
    """Extracted spatial gradient and orientation distributions for a patch."""
    luminance: np.ndarray  # (H, W) float in [0, 1]
    grad_x: np.ndarray  # Horizontal Sobel derivative
    grad_y: np.ndarray  # Vertical Sobel derivative
    magnitude: np.ndarray  # Gradient magnitude
    orientation: np.ndarray  # Gradient angle in radians [-pi, pi]
    mean_magnitude: float
    magnitude_std: float
    gradient_energy: float
    orientation_histogram: np.ndarray  # 16-bin normalized orientation distribution


class ChromaticDecoupler:
    """Decouples color temperature from structural brushwork and surface gradients."""

    def __init__(self, num_orientation_bins: int = 16) -> None:
        self.num_orientation_bins = num_orientation_bins

    def rgb_to_luminance(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Converts RGB array (H, W, 3) to standardized Rec. 709 luminance in range [0, 1].
        """
        arr = rgb_image.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0

        if arr.ndim == 3 and arr.shape[2] == 3:
            # ITU-R BT.709 luminance formula
            lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        elif arr.ndim == 2:
            lum = arr
        else:
            lum = arr[:, :, 0]

        return np.clip(lum, 0.0, 1.0)

    def compute_gradients(self, patch: np.ndarray) -> SpatialGradientFeatures:
        """
        Computes spatial derivatives, magnitude, and orientation distribution on a patch.
        """
        lum = self.rgb_to_luminance(patch)

        # Sobel convolution kernels
        gx = ndimage.sobel(lum, axis=1, mode="reflect")
        gy = ndimage.sobel(lum, axis=0, mode="reflect")

        mag = np.hypot(gx, gy)
        theta = np.arctan2(gy, gx)

        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))
        energy = float(np.sum(mag ** 2) / max(1, mag.size))

        # Weight orientations by magnitude (brushstroke orientation histogram)
        bins = np.linspace(-np.pi, np.pi, self.num_orientation_bins + 1)
        hist, _ = np.histogram(theta, bins=bins, weights=mag)
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist_norm = hist / hist_sum
        else:
            hist_norm = np.ones(self.num_orientation_bins, dtype=np.float32) / self.num_orientation_bins

        return SpatialGradientFeatures(
            luminance=lum,
            grad_x=gx,
            grad_y=gy,
            magnitude=mag,
            orientation=theta,
            mean_magnitude=mean_mag,
            magnitude_std=std_mag,
            gradient_energy=energy,
            orientation_histogram=hist_norm.astype(np.float32),
        )

    def extract_gradient_feature_vector(self, patch: np.ndarray) -> np.ndarray:
        """
        Constructs a compact 32-dimensional gradient feature descriptor for a patch:
        [16 orientation bins, 8 radial magnitude rings, mean, std, energy, skew, kurtosis, etc.]
        """
        feats = self.compute_gradients(patch)
        mag = feats.magnitude

        # Statistical moments of gradient magnitude
        mean = feats.mean_magnitude
        std = feats.magnitude_std
        energy = feats.gradient_energy

        centered = mag - mean
        skew = float(np.mean(centered ** 3) / max(1e-6, std ** 3))
        kurt = float(np.mean(centered ** 4) / max(1e-6, std ** 4))

        # Multi-scale radial energy (central vs peripheral gradients in patch)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        y_idx, x_idx = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist_norm = dist_from_center / max(1e-5, max_dist)

        radial_bins = 8
        radial_profile = np.zeros(radial_bins, dtype=np.float32)
        for i in range(radial_bins):
            r_min = i / radial_bins
            r_max = (i + 1) / radial_bins
            mask = (dist_norm >= r_min) & (dist_norm < r_max)
            if np.any(mask):
                radial_profile[i] = float(np.mean(mag[mask]))

        rad_sum = np.sum(radial_profile)
        if rad_sum > 0:
            radial_profile /= rad_sum

        scalar_stats = np.array([mean, std, energy, skew, kurt], dtype=np.float32)

        # Concatenate: 16 (orientation) + 8 (radial) + 5 (stats) + 3 (padding to 32)
        vec = np.concatenate([feats.orientation_histogram, radial_profile, scalar_stats, np.zeros(3, dtype=np.float32)])
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)
