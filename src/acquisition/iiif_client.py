"""
IIIF Manifest and Image API Client for Fine Art Collections.
Compliant with IIIF Image API 2.1 & 3.0 and Presentation API 2.0 & 3.0.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import requests

logger = logging.getLogger(__name__)


@dataclass
class IIIFTileSpec:
    """Specification for deep-zoom tiles defined in IIIF Image API info.json."""
    width: int
    height: Optional[int] = None
    scale_factors: List[int] = field(default_factory=lambda: [1])

    @property
    def tile_height(self) -> int:
        return self.height if self.height is not None else self.width


@dataclass
class IIIFImageService:
    """Represents a resolved IIIF Image Service endpoint."""
    service_url: str
    width: int
    height: int
    tiles: List[IIIFTileSpec] = field(default_factory=list)
    protocol: str = "http://iiif.io/api/image"
    profile: str = "http://iiif.io/api/image/2/level2.json"
    formats: List[str] = field(default_factory=lambda: ["jpg"])

    def construct_region_url(
        self,
        region: str | Tuple[int, int, int, int] = "full",
        size: str | Tuple[int, int] = "max",
        rotation: int | float = 0,
        quality: str = "default",
        output_format: str = "jpg",
    ) -> str:
        """
        Constructs standard IIIF Image API request URL.
        Format: {service_url}/{region}/{size}/{rotation}/{quality}.{format}
        """
        base = self.service_url.rstrip("/")

        if isinstance(region, tuple):
            rx, ry, rw, rh = region
            region_str = f"{rx},{ry},{rw},{rh}"
        else:
            region_str = str(region)

        if isinstance(size, tuple):
            sw, sh = size
            size_str = f"{sw},{sh}"
        else:
            size_str = str(size)

        rot_str = str(rotation)
        return f"{base}/{region_str}/{size_str}/{rot_str}/{quality}.{output_format}"


@dataclass
class IIIFManifestInfo:
    """Extracted metadata and canvas list from a IIIF Presentation Manifest."""
    manifest_url: str
    label: str
    artist: Optional[str] = None
    date: Optional[str] = None
    institution: Optional[str] = None
    canvases: List[Dict[str, Any]] = field(default_factory=list)
    primary_image_service: Optional[IIIFImageService] = None


class IIIFClient:
    """Client for querying and extracting artwork images from IIIF museum servers."""

    def __init__(self, session: Optional[requests.Session] = None, timeout: int = 20) -> None:
        self.session = session or requests.Session()
        self.timeout = timeout
        self.session.headers.update({
            "User-Agent": "ArtStylometry/0.1.0 (Computational Connoisseurship Engine; open-source research)"
        })

    def fetch_manifest(self, manifest_url: str) -> IIIFManifestInfo:
        """Fetches and parses a IIIF Presentation Manifest (v2 or v3)."""
        logger.info("Fetching IIIF manifest from: %s", manifest_url)
        resp = self.session.get(manifest_url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return self.parse_manifest(data, manifest_url)

    def parse_manifest(self, data: Dict[str, Any], manifest_url: str) -> IIIFManifestInfo:
        """Parses manifest JSON into structured IIIFManifestInfo."""
        label = self._extract_label(data.get("label", "Unknown Artwork"))
        artist = self._extract_metadata_field(data, ["artist", "creator", "painter", "author", "maker"])
        date = self._extract_metadata_field(data, ["date", "created", "period"])
        institution = self._extract_metadata_field(data, ["institution", "repository", "provider", "holding institution"])

        canvases: List[Dict[str, Any]] = []
        primary_service: Optional[IIIFImageService] = None

        # Presentation API v2: sequences[0].canvases
        sequences = data.get("sequences", [])
        if sequences and isinstance(sequences, list) and "canvases" in sequences[0]:
            for c in sequences[0].get("canvases", []):
                canvas_info = self._parse_v2_canvas(c)
                canvases.append(canvas_info)
                if not primary_service and "image_service" in canvas_info:
                    primary_service = canvas_info["image_service"]

        # Presentation API v3: items (canvases)
        items = data.get("items", [])
        if not canvases and items:
            for item in items:
                canvas_info = self._parse_v3_canvas(item)
                canvases.append(canvas_info)
                if not primary_service and "image_service" in canvas_info:
                    primary_service = canvas_info["image_service"]

        return IIIFManifestInfo(
            manifest_url=manifest_url,
            label=label,
            artist=artist,
            date=date,
            institution=institution,
            canvases=canvases,
            primary_image_service=primary_service,
        )

    def fetch_image_service_info(self, info_json_url: str) -> IIIFImageService:
        """Directly queries info.json from a IIIF Image API server."""
        if not info_json_url.endswith("info.json"):
            info_json_url = f"{info_json_url.rstrip('/')}/info.json"

        logger.info("Fetching IIIF image info from: %s", info_json_url)
        resp = self.session.get(info_json_url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        service_url = data.get("@id") or data.get("id") or info_json_url.rsplit("/info.json", 1)[0]
        width = int(data.get("width", 0))
        height = int(data.get("height", 0))

        tiles: List[IIIFTileSpec] = []
        for t in data.get("tiles", []):
            tw = int(t.get("width", 256))
            th = int(t.get("height")) if t.get("height") else tw
            sf = [int(s) for s in t.get("scaleFactors", [1])]
            tiles.append(IIIFTileSpec(width=tw, height=th, scale_factors=sf))

        if not tiles:
            tiles.append(IIIFTileSpec(width=512, height=512, scale_factors=[1, 2, 4, 8, 16]))

        return IIIFImageService(
            service_url=service_url,
            width=width,
            height=height,
            tiles=tiles,
            protocol=data.get("protocol", "http://iiif.io/api/image"),
            profile=str(data.get("profile", "")),
        )

    def _parse_v2_canvas(self, canvas: Dict[str, Any]) -> Dict[str, Any]:
        cid = canvas.get("@id", "")
        clabel = self._extract_label(canvas.get("label", ""))
        width = int(canvas.get("width", 0))
        height = int(canvas.get("height", 0))

        image_service = None
        images = canvas.get("images", [])
        if images and "resource" in images[0]:
            res = images[0]["resource"]
            svc = res.get("service", {})
            if isinstance(svc, dict) and ("@id" in svc or "id" in svc):
                s_url = svc.get("@id") or svc.get("id")
                image_service = IIIFImageService(
                    service_url=s_url,
                    width=width,
                    height=height,
                )

        return {
            "id": cid,
            "label": clabel,
            "width": width,
            "height": height,
            "image_service": image_service,
        }

    def _parse_v3_canvas(self, item: Dict[str, Any]) -> Dict[str, Any]:
        cid = item.get("id", "")
        clabel = self._extract_label(item.get("label", ""))
        width = int(item.get("width", 0))
        height = int(item.get("height", 0))

        image_service = None
        sub_items = item.get("items", [])
        if sub_items and "items" in sub_items[0]:
            annotation = sub_items[0]["items"][0]
            body = annotation.get("body", {})
            services = body.get("service", [])
            if services:
                target_svc = services[0] if isinstance(services, list) else services
                s_url = target_svc.get("id") or target_svc.get("@id")
                if s_url:
                    image_service = IIIFImageService(
                        service_url=s_url,
                        width=width,
                        height=height,
                    )

        return {
            "id": cid,
            "label": clabel,
            "width": width,
            "height": height,
            "image_service": image_service,
        }

    @staticmethod
    def _extract_label(raw: Any) -> str:
        if isinstance(raw, str):
            return raw
        if isinstance(raw, dict):
            # v3 internationalized string e.g. {"en": ["Title"], "fr": ["Titre"]}
            for val in raw.values():
                if isinstance(val, list) and val:
                    return str(val[0])
                if isinstance(val, str):
                    return val
        if isinstance(raw, list) and raw:
            return str(raw[0])
        return str(raw)

    @staticmethod
    def _extract_metadata_field(data: Dict[str, Any], candidate_keys: List[str]) -> Optional[str]:
        metadata = data.get("metadata", [])
        if isinstance(metadata, list):
            for entry in metadata:
                label = IIIFClient._extract_label(entry.get("label", "")).lower()
                for key in candidate_keys:
                    if key in label:
                        return IIIFClient._extract_label(entry.get("value", ""))
        return None
