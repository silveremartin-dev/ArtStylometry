"""
DINOv2 Native-Scale Texture Encoder.

Encodes *native-resolution* micro-crops (never the down-sampled canvas) with a
self-supervised DINOv2 ViT so that embeddings describe paint handling rather
than iconography. Crops are chosen deterministically on a regular grid and the
highest-variance ones are kept (blank background / unpainted ground is skipped).

The encoder degrades gracefully: if PyTorch or the hub weights are unavailable
(offline CI), ``available`` is False and callers fall back to analytical features.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_TORCH = False

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def select_native_crops(
    img: Image.Image,
    crop_size: int = 224,
    grid: int = 5,
    keep: int = 8,
    margin_frac: float = 0.06,
) -> List[np.ndarray]:
    """
    Samples a ``grid x grid`` lattice of native-resolution crops (excluding a frame margin
    that often contains frames, labels or colour charts) and keeps the ``keep`` crops with
    the highest luminance variance.
    """
    w, h = img.size
    if min(w, h) < crop_size:
        return [np.asarray(img.resize((crop_size, crop_size)).convert("RGB"))]
    mx, my = int(w * margin_frac), int(h * margin_frac)
    xs = np.linspace(mx, max(mx, w - mx - crop_size), grid).astype(int)
    ys = np.linspace(my, max(my, h - my - crop_size), grid).astype(int)
    crops: List[Tuple[float, np.ndarray]] = []
    for y in ys:
        for x in xs:
            c = np.asarray(img.crop((int(x), int(y), int(x) + crop_size, int(y) + crop_size)).convert("RGB"))
            lum = c.mean(axis=2)
            crops.append((float(lum.var()), c))
    crops.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in crops[:keep]]


class DinoTextureEncoder:
    """Batched DINOv2 CLS-token encoder for native-scale paint crops."""

    def __init__(self, model_name: str = "dinov2_vits14", device: Optional[str] = None, enabled: bool = True) -> None:
        self.model_name = model_name
        self.model = None
        self.device = "cpu"
        self.dim = 0
        if not (enabled and HAS_TORCH):
            return
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True, verbose=False)
            self.model.eval().to(self.device)
            self.dim = int(self.model.embed_dim)
            logger.info("DINOv2 %s loaded on %s (dim=%d)", model_name, self.device, self.dim)
        except Exception as e:  # offline / no weights
            logger.warning("DINOv2 unavailable (%s); analytical features only.", e)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def encode(self, crops: List[np.ndarray]) -> np.ndarray:
        """Returns L2-normalised (N, dim) CLS embeddings for 224x224 RGB uint8 crops."""
        if not self.available or not crops:
            return np.zeros((len(crops), self.dim), dtype=np.float32)
        arr = np.stack([c.astype(np.float32) / 255.0 for c in crops])
        arr = (arr - _MEAN) / _STD
        tensor = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(self.device)
        with torch.inference_mode():
            feats = self.model(tensor).cpu().numpy().astype(np.float32)
        feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        return feats
