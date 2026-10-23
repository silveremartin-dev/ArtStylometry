"""
High-Resolution Tile Downloader and Canvas Reconstructor for IIIF Images.
Supports downloading, rate-limiting, local caching, and seamless stitching.
"""

from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import requests

from .iiif_client import IIIFImageService

logger = logging.getLogger(__name__)


class TileDownloader:
    """Downloads tiles from a IIIF Image API endpoint and caches them locally."""

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        session: Optional[requests.Session] = None,
        timeout: int = 15,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.timeout = timeout

    def fetch_tile(
        self,
        service: IIIFImageService,
        region: Tuple[int, int, int, int],
        size: Tuple[int, int],
        output_format: str = "jpg",
    ) -> Image.Image:
        """
        Fetches an individual tile image, reading from cache if available.
        """
        rx, ry, rw, rh = region
        sw, sh = size

        cache_file: Optional[Path] = None
        if self.cache_dir:
            safe_name = service.service_url.replace("://", "_").replace("/", "_").replace(":", "_")
            tile_name = f"{safe_name}_{rx}_{ry}_{rw}_{rh}_{sw}x{sh}.{output_format}"
            cache_file = self.cache_dir / tile_name
            if cache_file.exists():
                try:
                    return Image.open(cache_file).convert("RGB")
                except Exception as e:
                    logger.warning("Corrupted cache tile %s: %s. Refetching.", cache_file, e)

        url = service.construct_region_url(
            region=region,
            size=size,
            output_format=output_format,
        )

        logger.debug("Requesting tile: %s", url)
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        if cache_file:
            try:
                img.save(cache_file)
            except Exception as e:
                logger.debug("Could not write cache file: %s", e)

        return img


class CanvasReconstructor:
    """Assembles full or sub-region high-resolution images from deep-zoom tiles."""

    def __init__(self, downloader: Optional[TileDownloader] = None) -> None:
        self.downloader = downloader or TileDownloader()

    def reconstruct_region(
        self,
        service: IIIFImageService,
        bounding_box: Optional[Tuple[int, int, int, int]] = None,
        max_dimension: Optional[int] = None,
    ) -> Image.Image:
        """
        Reconstructs a requested region of the artwork from tiles.
        If bounding_box is None, reconstructs the full canvas.
        """
        if bounding_box is None:
            bx, by, bw, bh = 0, 0, service.width, service.height
        else:
            bx, by, bw, bh = bounding_box

        # Constrain to image boundaries
        bx = max(0, min(bx, service.width - 1))
        by = max(0, min(by, service.height - 1))
        bw = min(bw, service.width - bx)
        bh = min(bh, service.height - by)

        # Select tile spec
        tile_w = 512
        tile_h = 512
        if service.tiles:
            tile_w = service.tiles[0].width
            tile_h = service.tiles[0].tile_height

        # Calculate scale factor if max_dimension is requested
        scale = 1
        if max_dimension:
            dominant_dim = max(bw, bh)
            if dominant_dim > max_dimension:
                scale = math.ceil(dominant_dim / max_dimension)

        eff_tile_w = tile_w * scale
        eff_tile_h = tile_h * scale

        out_w = math.ceil(bw / scale)
        out_h = math.ceil(bh / scale)

        logger.info(
            "Reconstructing canvas %s (%dx%d) -> target buffer (%dx%d) scale=%d",
            service.service_url,
            bw,
            bh,
            out_w,
            out_h,
            scale,
        )

        MAX_PIXELS = 100_000_000  # 100 MP limit to prevent Decompression Bomb / Out-Of-Memory DoS
        if out_w * out_h > MAX_PIXELS:
            raise ValueError(
                f"Requested canvas dimensions ({out_w}x{out_h} = {out_w * out_h} pixels) exceed safety limit of {MAX_PIXELS} pixels."
            )

        canvas = Image.new("RGB", (out_w, out_h), (0, 0, 0))

        y_steps = range(by, by + bh, eff_tile_h)
        x_steps = range(bx, bx + bw, eff_tile_w)

        for curr_y in y_steps:
            for curr_x in x_steps:
                rw = min(eff_tile_w, bx + bw - curr_x)
                rh = min(eff_tile_h, by + bh - curr_y)

                tw = math.ceil(rw / scale)
                th = math.ceil(rh / scale)

                tile_img = self.downloader.fetch_tile(
                    service=service,
                    region=(curr_x, curr_y, rw, rh),
                    size=(tw, th),
                )

                dest_x = math.floor((curr_x - bx) / scale)
                dest_y = math.floor((curr_y - by) / scale)

                # Ensure dimensions match tile before pasting
                if tile_img.size != (tw, th):
                    tile_img = tile_img.resize((tw, th), Image.Resampling.BILINEAR)

                canvas.paste(tile_img, (dest_x, dest_y))

        return canvas

    @staticmethod
    def load_local_canvas(file_path: str | Path) -> Image.Image:
        """Loads a local canvas image (TIFF, PNG, JPEG) and ensures RGB mode."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local canvas file not found: {path}")
        img = Image.open(path)
        return img.convert("RGB")
