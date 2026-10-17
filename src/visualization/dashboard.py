"""
Interactive Stylometry HTML Dashboard Generator.
Creates standalone, zero-dependency HTML reports for visual connoisseurship audits.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from ..attribution.heatmap_generator import CanvasHeatmapResult
from ..clustering.clusterer import ClusteringResult
from ..clustering.dimension_reduction import ManifoldProjection
from .heatmap_overlay import HeatmapOverlayRenderer


class StylometryDashboardGenerator:
    """Generates standalone visual HTML reports for stylistic attribution audits."""

    def __init__(self, renderer: Optional[HeatmapOverlayRenderer] = None) -> None:
        self.renderer = renderer or HeatmapOverlayRenderer()

    def generate_audit_report(
        self,
        artwork_id: str,
        original_image: Image.Image | np.ndarray,
        heatmap_result: CanvasHeatmapResult,
        output_html_path: str | Path,
    ) -> Path:
        """
        Creates an interactive HTML report for a single canvas attribution audit.
        """
        if isinstance(original_image, np.ndarray):
            orig_pil = Image.fromarray(original_image.astype(np.uint8)).convert("RGB")
        else:
            orig_pil = original_image.convert("RGB")

        overlay_pil = self.renderer.render_overlay(orig_pil, heatmap_result)

        orig_b64 = self._pil_to_base64(orig_pil, max_dim=800)
        overlay_b64 = self._pil_to_base64(overlay_pil, max_dim=800)

        conf = heatmap_result.overall_attribution_score
        status_label = (
            "Consensus Masterwork Hand" if conf >= 0.70
            else "Mixed Attribution / Studio Hand" if conf >= 0.40
            else "Statistical Stylistic Anomaly"
        )
        status_color = "#10b981" if conf >= 0.70 else "#f59e0b" if conf >= 0.40 else "#ef4444"

        zones_html = ""
        for z in heatmap_result.anomaly_zones:
            zones_html += f"""
            <tr>
                <td><strong>Zone {z.zone_id}</strong></td>
                <td>{z.bounding_box[0]}, {z.bounding_box[1]} ({z.bounding_box[2]}x{z.bounding_box[3]} px)</td>
                <td>{z.area_percentage:.1f}%</td>
                <td><span style="color: #ef4444; font-weight: bold;">{z.mean_confidence:.2f}</span></td>
                <td>{z.description}</td>
            </tr>
            """

        if not heatmap_result.anomaly_zones:
            zones_html = "<tr><td colspan='5' style='text-align:center; color:#9ca3af;'>No significant anomaly zones detected. Surface facture is uniform.</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ArtStylometry Audit: {artwork_id}</title>
    <style>
        body {{
            background: #0f172a;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 2rem;
        }}
        .header {{
            border-bottom: 1px solid #334155;
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        h1 {{ margin: 0; font-size: 1.8rem; color: #38bdf8; }}
        .badge {{
            background: {status_color};
            color: #0f172a;
            font-weight: bold;
            padding: 0.4rem 0.8rem;
            border-radius: 6px;
            font-size: 0.9rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: #1e293b;
            padding: 1.2rem;
            border-radius: 8px;
            border: 1px solid #334155;
        }}
        .metric-label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}
        .metric-value {{ font-size: 1.6rem; font-weight: bold; color: #f1f5f9; margin-top: 0.3rem; }}
        .image-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .image-card {{
            background: #1e293b;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #334155;
            text-align: center;
        }}
        .image-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .image-title {{ font-weight: 600; margin-bottom: 0.8rem; color: #cbd5e1; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #334155;
        }}
        th, td {{
            padding: 0.8rem 1rem;
            text-align: left;
            border-bottom: 1px solid #334155;
        }}
        th {{ background: #0f172a; color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>ArtStylometry Connoisseurship Audit</h1>
            <p style="margin: 0.3rem 0 0 0; color: #94a3b8;">Artwork ID: <strong>{artwork_id}</strong> &bull; Target Artist Profile: <strong>{heatmap_result.artist_name}</strong></p>
        </div>
        <div class="badge">{status_label}</div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Overall Confidence</div>
            <div class="metric-value">{conf * 100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Patches Evaluated</div>
            <div class="metric-value">{heatmap_result.num_evaluated_patches}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Canvas Dimensions</div>
            <div class="metric-value">{heatmap_result.canvas_dimensions[0]} &times; {heatmap_result.canvas_dimensions[1]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Anomaly Zones Detected</div>
            <div class="metric-value">{len(heatmap_result.anomaly_zones)}</div>
        </div>
    </div>

    <div class="image-container">
        <div class="image-card">
            <div class="image-title">1. Original Artwork Canvas</div>
            <img src="data:image/jpeg;base64,{orig_b64}" alt="Original Canvas">
        </div>
        <div class="image-card">
            <div class="image-title">2. Stylistic Attribution Heatmap Overlay</div>
            <img src="data:image/jpeg;base64,{overlay_b64}" alt="Attribution Heatmap Overlay">
        </div>
    </div>

    <h2 style="color: #38bdf8; font-size: 1.3rem;">Spatial Facture Anomaly Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Zone</th>
                <th>Bounding Box (x, y, w, h)</th>
                <th>Surface Area</th>
                <th>Local Confidence</th>
                <th>Interpretive Analysis</th>
            </tr>
        </thead>
        <tbody>
            {zones_html}
        </tbody>
    </table>
</body>
</html>
"""
        out_path = Path(output_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html_content, encoding="utf-8")
        return out_path

    @staticmethod
    def _pil_to_base64(img: Image.Image, max_dim: int = 800) -> str:
        """Scales image down if necessary and converts to JPEG base64 string."""
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")
