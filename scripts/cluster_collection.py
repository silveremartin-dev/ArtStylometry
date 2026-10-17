"""
Collection Clustering CLI.
Extracts dual-scale descriptors across an artwork collection and groups them
into unsupervised stylistic clusters using HDBSCAN and UMAP/PCA projection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import ArtStylometryPipeline


def main():
    parser = argparse.ArgumentParser(description="Cluster a collection of artworks into stylistic schools")
    parser.add_argument("--images", required=True, nargs="+", help="Paths to paintings to cluster")
    parser.add_argument("--min-cluster-size", type=int, default=3, help="Minimum cluster size")
    parser.add_argument("--deep", action="store_true", help="Attempt to use DINOv2 deep vision transformer")

    args = parser.parse_args()

    pipeline = ArtStylometryPipeline(use_deep_model=args.deep)
    cluster_res, proj, _ = pipeline.cluster_collection(args.images, args.min_cluster_size)

    print("=" * 60)
    print("COLLECTION CLUSTERING RESULTS")
    print("=" * 60)
    print(f"Total Artworks Evaluated : {len(cluster_res.artwork_ids)}")
    print(f"Discovered Clusters     : {cluster_res.num_clusters}")
    print(f"Outliers / Noise Works   : {cluster_res.num_outliers} ({cluster_res.outlier_percentage:.1f}%)")
    print(f"Method Used             : {cluster_res.method_used} + {proj.method_used}")
    if cluster_res.silhouette is not None:
        print(f"Silhouette Score        : {cluster_res.silhouette:.3f}")
    print("-" * 60)
    print("Artwork Memberships:")
    for aid, lbl, prob, coord in zip(
        cluster_res.artwork_ids,
        cluster_res.labels,
        cluster_res.probabilities,
        proj.coordinates,
    ):
        status = "CORE" if lbl != -1 else "OUTLIER"
        print(f"  [{status}] Cluster {lbl:2d} (p={prob:.2f}) | 2D: ({coord[0]:6.2f}, {coord[1]:6.2f}) | {aid}")


if __name__ == "__main__":
    main()
