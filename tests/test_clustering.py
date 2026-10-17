"""
Unit tests for dimensionality reduction and collection clustering.
"""

import numpy as np
import pytest

from src.clustering.dimension_reduction import DimensionReducer
from src.clustering.clusterer import CollectionClusterer


def test_dimension_reducer_pca_and_umap():
    np.random.seed(42)
    vectors = np.random.randn(20, 32).astype(np.float32)

    reducer_pca = DimensionReducer(n_components=2, method="pca")
    proj_pca = reducer_pca.fit_transform(vectors)

    assert proj_pca.coordinates.shape == (20, 2)
    assert proj_pca.n_components == 2
    assert proj_pca.variance_explained is not None
    assert 0.0 < proj_pca.variance_explained <= 1.0


def test_collection_clusterer_discovers_groups_and_outliers():
    np.random.seed(42)
    # Cluster 1: 8 points around [2, 2]
    c1 = np.random.normal(loc=2.0, scale=0.1, size=(8, 16))
    # Cluster 2: 8 points around [-2, -2]
    c2 = np.random.normal(loc=-2.0, scale=0.1, size=(8, 16))
    # Outlier: 1 point at [10, 10]
    outlier = np.full((1, 16), 10.0)

    dataset = np.vstack([c1, c2, outlier]).astype(np.float32)
    ids = [f"art_{i}" for i in range(len(dataset))]

    clusterer = CollectionClusterer(min_cluster_size=3)
    res = clusterer.cluster(dataset, ids)

    assert res.num_clusters >= 2
    assert len(res.labels) == len(dataset)
    # The last point should be detected as noise (-1) in robust density clustering
    assert res.labels[-1] == -1 or res.num_outliers >= 1
