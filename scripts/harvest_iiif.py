"""
IIIF Museum Painting Harvester.
Downloads high-resolution paintings from open-access museum IIIF APIs
(Rijksmuseum, National Gallery London, Art Institute of Chicago, The Met, Gallica).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition.iiif_client import IIIFClient
from src.acquisition.downloader import TileDownloader, CanvasReconstructor


def harvest_artwork(
    manifest_url_or_info: str,
    output_path: str,
    max_dimension: int = 4000,
) -> None:
    print(f"[*] Querying IIIF endpoint: {manifest_url_or_info}")
    client = IIIFClient()
    downloader = TileDownloader(cache_dir="data/cache/tiles")
    reconstructor = CanvasReconstructor(downloader)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if manifest_url_or_info.endswith("info.json"):
        service = client.fetch_image_service_info(manifest_url_or_info)
        print(f"[*] Native resolution: {service.width} x {service.height}")
    else:
        manifest_info = client.fetch_manifest(manifest_url_or_info)
        print(f"[*] Manifest: '{manifest_info.label}' by {manifest_info.artist or 'Unknown'}")
        if not manifest_info.primary_image_service:
            raise RuntimeError("No image service found in manifest")
        service = manifest_info.primary_image_service
        print(f"[*] Resolved image service: {service.service_url} ({service.width}x{service.height})")

    canvas = reconstructor.reconstruct_region(service, max_dimension=max_dimension)
    canvas.save(out_file)
    print(f"[+] Successfully harvested canvas to: {out_file} ({canvas.size[0]}x{canvas.size[1]} px)")


def main():
    parser = argparse.ArgumentParser(description="Harvest high-resolution museum paintings via IIIF")
    parser.add_argument("--url", required=True, help="IIIF Presentation Manifest URL or Image API info.json URL")
    parser.add_argument("--out", required=True, help="Destination file path (.jpg, .png, .tif)")
    parser.add_argument("--max-dimension", type=int, default=4000, help="Max width/height limit (preserves deep zoom)")

    args = parser.parse_args()
    harvest_artwork(args.url, args.out, args.max_dimension)


if __name__ == "__main__":
    main()
