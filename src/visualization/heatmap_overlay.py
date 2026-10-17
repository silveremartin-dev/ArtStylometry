"""
Attribution Heatmap Overlay Renderer.
Blends confidence heatmaps onto original artwork images and annotates detected studio anomaly zones.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..attribution.heatmap_generator import CanvasHeatmapResult, AnomalyZone


class HeatmapOverlayRenderer:
    """Renders attribution heatmaps blended on original canvases with zone annotations."""

    def __init__(self, colormap_name: str = "inferno", alpha: float = 0.55) -> None:
        self.colormap_name = colormap_name
        self.alpha = alpha

    def render_overlay(
        self,
        original_image: Image.Image | np.ndarray,
        heatmap_result: CanvasHeatmapResult,
        highlight_anomalies: bool = True,
    ) -> Image.Image:
        """
        Blends original image with colored confidence heatmap and outlines anomaly zones.
        """
        if isinstance(original_image, np.ndarray):
            orig_pil = Image.fromarray(original_image.astype(np.uint8)).convert("RGB")
        else:
            orig_pil = original_image.convert("RGB")

        orig_w, orig_h = orig_pil.size
        heatmap = heatmap_result.heatmap

        # Resize heatmap to match original image dimensions if necessary
        if heatmap.shape != (orig_h, orig_w):
            heatmap_pil = Image.fromarray((heatmap * 255.0).astype(np.uint8))
            resized_h = heatmap_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR)
            h_arr = np.array(resized_h, dtype=np.float32) / 255.0
        else:
            h_arr = heatmap

        # Map scalar values [0, 1] to RGBA colormap
        try:
            import matplotlib
            cmap = matplotlib.colormaps[self.colormap_name]
        except Exception:
            cmap = cm.get_cmap(self.colormap_name)
        colored = cmap(h_arr)  # (H, W, 4) in [0, 1]
        colored_rgb = (colored[:, :, :3] * 255.0).astype(np.uint8)
        heatmap_overlay = Image.fromarray(colored_rgb, mode="RGB")

        # Alpha blend original image and heatmap overlay
        blended = Image.blend(orig_pil, heatmap_overlay, alpha=self.alpha)

        # Highlight anomaly zones if requested
        if highlight_anomalies and heatmap_result.anomaly_zones:
            draw = ImageDraw.Draw(blended)
            for zone in heatmap_result.anomaly_zones:
                zx, zy, zw, zh = zone.bounding_box
                # Draw high-visibility rectangle outline
                draw.rectangle([zx, zy, zx + zw, zy + zh], outline=(255, 50, 50), width=3)
                label_text = f"Zone {zone.zone_id}: {zone.area_percentage:.1f}% (conf: {zone.mean_confidence:.2f})"
                # Draw text background
                draw.rectangle([zx, zy - 18, zx + len(label_text) * 8, zy], fill=(0, 0, 0, 180))
                draw.text((zx + 4, zy - 16), label_text, fill=(255, 255, 255))

        return blended

    def save_side_by_side(
        self,
        original_image: Image.Image | np.ndarray,
        heatmap_result: CanvasHeatmapResult,
        output_path: str | Path,
    ) -> None:
        """Saves a 2-panel comparison: [Original Artwork | Blended Attribution Heatmap]."""
        if isinstance(original_image, np.ndarray):
            orig_pil = Image.fromarray(original_image.astype(np.uint8)).convert("RGB")
        else:
            orig_pil = original_image.convert("RGB")

        overlay = self.render_overlay(orig_pil, heatmap_result)
        w, h = orig_pil.size

        composite = Image.new("RGB", (w * 2, h), (0, 0, 0))
        composite.paste(orig_pil, (0, 0))
        composite.paste(overlay, (w, 0))

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        composite.save(out)
