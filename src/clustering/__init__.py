"""
Unsupervised Collection Clustering Engine.
Projects artwork descriptors into low-dimensional manifolds (UMAP/PCA)
and clusters collections via density algorithms (HDBSCAN/DBSCAN).
"""

from .dimension_reduction import DimensionReducer, ManifoldProjection
from .clusterer import CollectionClusterer, ClusteringResult

__all__ = [
    "DimensionReducer",
    "ManifoldProjection",
    "CollectionClusterer",
    "ClusteringResult",
]
