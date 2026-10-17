"""
Frequency Bandpass and Facture Isolation Filter.
Mitigates the craquelure and varnish confounding trap by isolating
the spatial frequencies corresponding to authentic painterly brushwork (2-8 cycles/mm)
while suppressing high-frequency crack edges and low-frequency varnish gradients.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


class FactureBandpassFilter:
    """
    Applies spatial frequency bandpass filtering via Difference of Gaussians (DoG)
    or 2D Fourier filtering to decouple brushwork facture from craquelure and varnish.
    """

    def __init__(
        self,
        low_sigma: float = 1.2,
        high_sigma: float = 8.0,
    ) -> None:
        """
        Args:
            low_sigma: Standard deviation for small Gaussian kernel (suppresses micro-cracks/noise).
            high_sigma: Standard deviation for large Gaussian kernel (suppresses varnish gradients).
        """
        if low_sigma >= high_sigma:
            raise ValueError(f"low_sigma ({low_sigma}) must be less than high_sigma ({high_sigma})")
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma

    def filter_luminance(self, luminance: np.ndarray) -> np.ndarray:
        """
        Extracts pure painterly facture signal by computing Difference of Gaussians:
            Facture(x, y) = G_{low_sigma}(I) - G_{high_sigma}(I)
        
        Args:
            luminance: 2D float32 array in [0, 1].
            
        Returns:
            2D float32 bandpass-filtered facture array normalized in [0, 1].
        """
        lum = luminance.astype(np.float32)
        if lum.max() > 1.0:
            lum /= 255.0

        # Small blur suppresses micro-craquelures and single-pixel sensor noise
        smoothed_micro = ndimage.gaussian_filter(lum, sigma=self.low_sigma)

        # Large blur captures low-frequency illumination, varnish discoloration, and canvas warping
        smoothed_macro = ndimage.gaussian_filter(lum, sigma=self.high_sigma)

        # Bandpass isolates physical bristle strokes and impasto ridges
        bandpass = smoothed_micro - smoothed_macro

        # Rescale bandpass to [0, 1] range centered at 0.5
        norm_bandpass = (bandpass - bandpass.min()) / (bandpass.max() - bandpass.min() + 1e-8)
        return norm_bandpass.astype(np.float32)

    def extract_facture_spectral_energy(self, patch: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyzes 2D FFT spectral energy distribution of an image patch.
        
        Returns:
            Tuple of (low_freq_ratio, mid_freq_facture_ratio, high_freq_crack_ratio).
        """
        if patch.ndim == 3:
            lum = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
        else:
            lum = patch.astype(np.float32)

        h, w = lum.shape
        fft = np.fft.fftshift(np.fft.fft2(lum))
        magnitude = np.abs(fft) ** 2

        # Radial distance matrix from center (DC)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        max_r = np.sqrt(cx**2 + cy**2)
        r_low = max_r * 0.08
        r_high = max_r * 0.40

        low_mask = r < r_low
        mid_mask = (r >= r_low) & (r < r_high)
        high_mask = r >= r_high

        total_energy = np.sum(magnitude) + 1e-8
        low_energy = np.sum(magnitude[low_mask]) / total_energy
        mid_energy = np.sum(magnitude[mid_mask]) / total_energy
        high_energy = np.sum(magnitude[high_mask]) / total_energy

        return float(low_energy), float(mid_energy), float(high_energy)

    extract_spectral_energy = extract_facture_spectral_energy
