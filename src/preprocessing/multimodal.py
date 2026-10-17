"""
Multimodal Imaging Integration for Fine Art Forensic Analysis.
Mitigates the 2D visible-light limitation by fusing Visible RGB,
Infrared Reflectography (IRR), X-Ray Radiography (XRR),
and Ultraviolet Fluorescence (UVF) layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Supported technical and forensic imaging modalities."""
    VIS_RGB = "vis_rgb"      # Visible Light RGB (surface facture)
    IRR = "irr"              # Infrared Reflectography (carbon underdrawing, pentimenti)
    XRR = "xrr"              # X-Ray Radiography (lead densities, support structure)
    UVF = "uvf"              # Ultraviolet Fluorescence (varnish state, modern retouches)
    XRF_MAP = "xrf_map"      # Macro-XRF Elemental Distribution Map (pigment chemistry)


@dataclass
class ModalityLayer:
    """Represents a single registered imaging modality for an artwork."""
    modality: ModalityType
    image_data: np.ndarray  # 2D grayscale or 3D RGB/multichannel array in float32 [0, 1]
    resolution_dpi: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        if self.image_data.dtype != np.float32:
            self.image_data = self.image_data.astype(np.float32)
            if self.image_data.max() > 1.0:
                self.image_data /= 255.0


@dataclass
class MultimodalArtwork:
    """
    Composite multimodal representation of an artwork combining surface photography
    with subsurface structural and forensic diagnostics.
    """
    artwork_id: str
    layers: Dict[ModalityType, ModalityLayer] = field(default_factory=dict)

    def add_layer(
        self,
        modality: ModalityType,
        image: np.ndarray | Image.Image,
        resolution_dpi: Optional[float] = None,
        description: str = "",
    ) -> None:
        """Adds a registered modality layer to the artwork stack."""
        if isinstance(image, Image.Image):
            arr = np.array(image, dtype=np.float32) / 255.0
        else:
            arr = image.astype(np.float32)
            if arr.max() > 1.0:
                arr /= 255.0

        # Convert RGB to single-channel luminance if technical modality (IRR, XRR, UVF)
        if modality in (ModalityType.IRR, ModalityType.XRR) and arr.ndim == 3:
            # Luminance weights (Rec. 709)
            arr = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

        self.layers[modality] = ModalityLayer(
            modality=modality,
            image_data=arr,
            resolution_dpi=resolution_dpi,
            description=description,
        )

    def has_subsurface_diagnostics(self) -> bool:
        """Returns True if any non-visible subsurface diagnostic is present."""
        return any(
            m in self.layers
            for m in (ModalityType.IRR, ModalityType.XRR, ModalityType.XRF_MAP)
        )

    def compute_cross_layer_coherence(
        self,
        patch_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, float]:
        """
        Computes spatial gradient correlation between visible surface paint (VIS)
        and subsurface underdrawings (IRR) or structural impasto (XRR).

        A high correlation indicates that surface brushwork follows the master's
        authentic preparatory underdrawing. A severe drop indicates late overpaints
        or studio assistant divergent execution.
        """
        vis_layer = self.layers.get(ModalityType.VIS_RGB)
        if vis_layer is None:
            return {"coherence_score": 1.0, "status": "no_visible_layer"}

        vis_img = vis_layer.image_data
        if vis_img.ndim == 3:
            vis_lum = 0.2126 * vis_img[:, :, 0] + 0.7152 * vis_img[:, :, 1] + 0.0722 * vis_img[:, :, 2]
        else:
            vis_lum = vis_img

        results: Dict[str, float] = {}

        # 1. Underdrawing correlation (IRR)
        irr_layer = self.layers.get(ModalityType.IRR)
        if irr_layer is not None:
            irr_img = irr_layer.image_data
            if irr_img.shape == vis_lum.shape:
                results["irr_coherence"] = self._gradient_correlation(vis_lum, irr_img, patch_box)

        # 2. Structural density correlation (XRR)
        xrr_layer = self.layers.get(ModalityType.XRR)
        if xrr_layer is not None:
            xrr_img = xrr_layer.image_data
            if xrr_img.shape == vis_lum.shape:
                results["xrr_coherence"] = self._gradient_correlation(vis_lum, xrr_img, patch_box)

        # 3. Retouching / varnish disturbance index (UVF)
        uvf_layer = self.layers.get(ModalityType.UVF)
        if uvf_layer is not None:
            uvf_img = uvf_layer.image_data
            if uvf_img.shape[:2] == vis_lum.shape:
                # High dark absorption patches under UV indicate modern retouching
                if uvf_img.ndim == 3:
                    uvf_dark = 1.0 - uvf_img.mean(axis=2)
                else:
                    uvf_dark = 1.0 - uvf_img
                results["retouch_risk_index"] = float(np.mean(uvf_dark > 0.75))

        # Overall composite confidence multiplier
        mult = 1.0
        if "irr_coherence" in results:
            mult *= (0.5 + 0.5 * max(0.0, results["irr_coherence"]))
        if "retouch_risk_index" in results:
            mult *= (1.0 - 0.5 * results["retouch_risk_index"])

        results["composite_multimodal_multiplier"] = float(np.clip(mult, 0.1, 1.0))
        return results

    @staticmethod
    def _gradient_correlation(
        img1: np.ndarray,
        img2: np.ndarray,
        box: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """Computes Pearson correlation between 2D gradient magnitudes of two layers."""
        if box:
            x, y, w, h = box
            sub1 = img1[y : y + h, x : x + w]
            sub2 = img2[y : y + h, x : x + w]
        else:
            sub1, sub2 = img1, img2

        # Sobel gradients
        gy1, gx1 = np.gradient(sub1)
        mag1 = np.sqrt(gx1**2 + gy1**2)

        gy2, gx2 = np.gradient(sub2)
        mag2 = np.sqrt(gx2**2 + gy2**2)

        m1_flat = mag1.ravel() - np.mean(mag1)
        m2_flat = mag2.ravel() - np.mean(mag2)

        std1 = np.std(mag1)
        std2 = np.std(mag2)

        if std1 < 1e-7 or std2 < 1e-7:
            return 0.5  # Neutral fallback

        corr = np.dot(m1_flat, m2_flat) / (len(m1_flat) * std1 * std2)
        return float(np.clip(corr, -1.0, 1.0))
