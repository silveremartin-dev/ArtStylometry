"""
ArtStylometry Central Pipeline & CLI Orchestrator.
Unifies acquisition, preprocessing, embedding extraction, attribution heatmaps, and collection clustering.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

from .acquisition.iiif_client import IIIFClient
from .acquisition.downloader import CanvasReconstructor, TileDownloader
from .preprocessing.tiler import ArtworkTiler
from .embeddings.micro_extractor import MicroFactureExtractor
from .embeddings.macro_extractor import MacroCompositionExtractor
from .embeddings.aggregator import PatchAggregator, ArtworkDescriptor
from .attribution.reference_profile import ArtistReferenceProfile, DistanceMetric
from .attribution.heatmap_generator import AttributionHeatmapGenerator
from .attribution.anomaly_detector import StylisticAnomalyDetector
from .clustering.dimension_reduction import DimensionReducer
from .clustering.clusterer import CollectionClusterer
from .visualization.heatmap_overlay import HeatmapOverlayRenderer
from .visualization.dashboard import StylometryDashboardGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ArtStylometry")


class ArtStylometryPipeline:
    """End-to-end computational connoisseurship and stylometry pipeline."""

    def __init__(
        self,
        use_deep_model: bool = False,
        patch_size: int = 256,
        stride: int = 128,
        cache_dir: str = "data/cache",
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.micro_extractor = MicroFactureExtractor(use_deep_model=use_deep_model)
        self.macro_extractor = MacroCompositionExtractor()
        self.aggregator = PatchAggregator(include_macro=True)
        self.tiler = ArtworkTiler(patch_size=patch_size, stride=stride)
        self.heatmap_generator = AttributionHeatmapGenerator(
            micro_extractor=self.micro_extractor,
            patch_size=patch_size,
            stride=stride,
        )
        self.overlay_renderer = HeatmapOverlayRenderer()
        self.dashboard_generator = StylometryDashboardGenerator(self.overlay_renderer)
        self.iiif_client = IIIFClient()
        self.reconstructor = CanvasReconstructor(TileDownloader(cache_dir=self.cache_dir / "tiles"))

    def build_artist_profile(
        self,
        artist_name: str,
        image_paths: List[str | Path],
    ) -> ArtistReferenceProfile:
        """
        Builds a reference profile from a set of uncontested paintings.
        """
        logger.info("Building reference profile for '%s' from %d artworks...", artist_name, len(image_paths))
        all_patch_vectors: List[np.ndarray] = []

        for p in image_paths:
            path = Path(p)
            if not path.exists():
                logger.warning("Artwork file not found: %s", path)
                continue
            img = Image.open(path).convert("RGB")
            grid = self.tiler.slice_image(img)
            for patch, _ in grid:
                res = self.micro_extractor.extract_patch(patch)
                if res.is_valid and res.cleanliness > 0.4:
                    all_patch_vectors.append(res.vector)

        if not all_patch_vectors:
            raise ValueError(f"No valid patches extracted from reference artworks for {artist_name}")

        stacked = np.stack(all_patch_vectors, axis=0)
        profile = ArtistReferenceProfile.fit_from_patch_vectors(
            artist_name=artist_name,
            patch_vectors=stacked,
            num_artworks=len(image_paths),
        )
        logger.info(
            "Reference profile fitted: %d patches, median distance: %.3f",
            profile.num_reference_patches,
            profile.median_distance,
        )
        return profile

    def audit_artwork(
        self,
        artwork_path: str | Path,
        reference_profile: ArtistReferenceProfile,
        output_dir: str | Path = "data/processed/audits",
    ) -> Path:
        """
        Performs an intra-canvas attribution audit and exports side-by-side overlay & HTML dashboard.
        """
        path = Path(artwork_path)
        img = Image.open(path).convert("RGB")
        art_id = path.stem

        out_dir = Path(output_dir) / art_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Auditing '%s' against reference profile '%s'...", art_id, reference_profile.artist_name)
        heatmap_res = self.heatmap_generator.generate(img, reference_profile)

        # Save side-by-side visualization
        sbs_path = out_dir / f"{art_id}_stylometric_heatmap.jpg"
        self.overlay_renderer.save_side_by_side(img, heatmap_res, sbs_path)

        # Generate HTML report
        html_path = out_dir / "index.html"
        self.dashboard_generator.generate_audit_report(art_id, img, heatmap_res, html_path)

        logger.info("Audit completed. Report: %s (Overall confidence: %.1f%%)", html_path, heatmap_res.overall_attribution_score * 100)
        return html_path

    def process_artwork_to_descriptor(
        self,
        artwork_path: str | Path,
    ) -> ArtworkDescriptor:
        """Extracts unified artwork descriptor combining micro-textures and macro-composition."""
        path = Path(artwork_path)
        img = Image.open(path).convert("RGB")
        art_id = path.stem

        grid = self.tiler.slice_image(img)
        patch_results = [self.micro_extractor.extract_patch(p, m.patch_id) for p, m in grid]
        macro_feats = self.macro_extractor.extract(img)

        return self.aggregator.aggregate(
            patch_results=patch_results,
            artwork_id=art_id,
            composition=macro_feats,
        )

    def cluster_collection(
        self,
        artwork_paths: List[str | Path],
        min_cluster_size: int = 3,
    ) -> tuple:
        """
        Extracts descriptors across a collection and groups them into unsupervised stylistic clusters.
        """
        logger.info("Extracting descriptors for %d artworks...", len(artwork_paths))
        descriptors: List[ArtworkDescriptor] = []
        for p in artwork_paths:
            try:
                desc = self.process_artwork_to_descriptor(p)
                descriptors.append(desc)
            except Exception as e:
                logger.warning("Failed to process %s: %s", p, e)

        if not descriptors:
            raise ValueError("No artworks successfully processed for clustering")

        vectors = np.stack([d.feature_vector for d in descriptors], axis=0)
        ids = [d.artwork_id for d in descriptors]

        clusterer = CollectionClusterer(min_cluster_size=min_cluster_size)
        cluster_res = clusterer.cluster(vectors, ids)

        reducer = DimensionReducer(n_components=2)
        proj = reducer.fit_transform(vectors)

        return cluster_res, proj, descriptors


def main() -> None:
    """Command-line entrypoint for ArtStylometry."""
    parser = argparse.ArgumentParser(
        description="ArtStylometry: Computational Stylometry & Quantitative Connoisseurship Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit an artwork against a reference corpus")
    audit_parser.add_argument("--artwork", required=True, help="Path to test painting image")
    audit_parser.add_argument("--references", required=True, nargs="+", help="Paths to canonical reference paintings")
    audit_parser.add_argument("--artist", default="Master", help="Name of reference artist")
    audit_parser.add_argument("--output-dir", default="data/processed/audits", help="Output directory")

    # Build browser command
    browser_parser = subparsers.add_parser("build-browser", help="Compile standalone interactive artwork browser HTML")
    browser_parser.add_argument("--output", default="data/processed/browser.html", help="Destination HTML file path")

    # Cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Cluster a collection of artworks")
    cluster_parser.add_argument("--images", required=True, nargs="+", help="Paths to painting images")
    cluster_parser.add_argument("--min-cluster-size", type=int, default=3, help="Minimum cluster size")

    args = parser.parse_args()

    pipeline = ArtStylometryPipeline()

    if args.command == "build-browser":
        from .browser.renderer import ArtworkBrowserRenderer
        renderer = ArtworkBrowserRenderer()
        out_path = renderer.export(args.output)
        print(f"[OK] Compiled interactive artwork browser to: {out_path}")
    elif args.command == "audit":
        profile = pipeline.build_artist_profile(args.artist, args.references)
        report = pipeline.audit_artwork(args.artwork, profile, args.output_dir)
        print(f"Report written to: {report}")
    elif args.command == "cluster":
        res, proj, _ = pipeline.cluster_collection(args.images, args.min_cluster_size)
        print(f"Found {res.num_clusters} clusters, {res.num_outliers} outliers ({res.outlier_percentage:.1f}%)")
        for aid, lbl in zip(res.artwork_ids, res.labels):
            print(f"  [{lbl}] {aid}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
