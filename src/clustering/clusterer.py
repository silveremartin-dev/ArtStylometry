"""
Density-Based Artwork Collection Clusterer.
Groups collections into stylistic schools / workshops using HDBSCAN / DBSCAN
and isolates controversial paintings into the statistical noise class (-1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Check for hdbscan library availability
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from sklearn.cluster import HDBSCAN as SkHDBSCAN
    HAS_SK_HDBSCAN = True
except ImportError:  # scikit-learn < 1.3
    HAS_SK_HDBSCAN = False


@dataclass
class ClusteringResult:
    """Clustering outcome for a collection of artworks."""
    artwork_ids: List[str]
    labels: List[int]  # Cluster IDs; -1 indicates noise / outlier
    probabilities: List[float]  # Membership strength [0, 1]
    num_clusters: int
    num_outliers: int
    outlier_percentage: float
    silhouette: Optional[float]
    method_used: str
    cluster_member_counts: Dict[int, int] = field(default_factory=dict)


class CollectionClusterer:
    """Performs density clustering on high-dimensional artwork descriptors."""

    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        allow_noise: bool = True,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or max(1, min_cluster_size // 2)
        self.metric = metric
        self.allow_noise = allow_noise

    def cluster(
        self,
        artwork_vectors: np.ndarray,
        artwork_ids: Optional[List[str]] = None,
    ) -> ClusteringResult:
        """
        Clusters artwork descriptors. Outliers are assigned label -1.
        """
        vectors = np.asarray(artwork_vectors, dtype=np.float32)
        n_samples = len(vectors)

        if artwork_ids is None:
            artwork_ids = [f"artwork_{i:03d}" for i in range(n_samples)]

        if n_samples < 2:
            return ClusteringResult(
                artwork_ids=artwork_ids,
                labels=[0] * n_samples,
                probabilities=[1.0] * n_samples,
                num_clusters=1,
                num_outliers=0,
                outlier_percentage=0.0,
                silhouette=None,
                method_used="trivial",
                cluster_member_counts={0: n_samples},
            )

        labels: np.ndarray
        probs: np.ndarray
        method: str

        eff_min_cluster = min(self.min_cluster_size, max(2, n_samples // 3))

        # Attempt HDBSCAN (standalone package first, then scikit-learn's native implementation)
        if HAS_HDBSCAN and n_samples >= 4:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=eff_min_cluster,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    prediction_data=True,
                )
                labels = clusterer.fit_predict(vectors)
                probs = clusterer.probabilities_
                method = "hdbscan"
            except Exception as e:
                logger.warning("HDBSCAN failed (%s). Falling back to DBSCAN.", e)
                labels, probs, method = self._run_dbscan(vectors)
        elif HAS_SK_HDBSCAN and n_samples >= 4:
            sk = SkHDBSCAN(
                min_cluster_size=eff_min_cluster,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method="leaf" if n_samples >= 200 else "eom",
            )
            labels = sk.fit_predict(vectors)
            probs = sk.probabilities_
            method = "sklearn_hdbscan"
        else:
            labels, probs, method = self._run_dbscan(vectors)

        # Count clusters & outliers
        unique_labels = set(labels)
        num_clusters = len([l for l in unique_labels if l != -1])
        num_outliers = int(np.sum(labels == -1))
        outlier_pct = float((num_outliers / n_samples) * 100.0)

        counts: Dict[int, int] = {}
        for l in labels:
            counts[int(l)] = counts.get(int(l), 0) + 1

        # Calculate silhouette score if > 1 valid cluster exists
        sil: Optional[float] = None
        if num_clusters > 1 and len(set(labels[labels != -1])) > 1:
            try:
                valid_mask = labels != -1
                if np.sum(valid_mask) > num_clusters:
                    sil = float(silhouette_score(vectors[valid_mask], labels[valid_mask]))
            except Exception:
                sil = None

        return ClusteringResult(
            artwork_ids=artwork_ids,
            labels=[int(l) for l in labels],
            probabilities=[float(p) for p in probs],
            num_clusters=num_clusters,
            num_outliers=num_outliers,
            outlier_percentage=outlier_pct,
            silhouette=sil,
            method_used=method,
            cluster_member_counts=counts,
        )

    def _run_dbscan(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        """Runs DBSCAN as a robust fallback."""
        # Calculate dynamic eps based on pairwise distances
        from scipy.spatial.distance import pdist
        if len(vectors) > 1:
            dists = pdist(vectors, metric=self.metric)
            eps = max(0.1, float(np.percentile(dists, 25)))
        else:
            eps = 0.5

        db = DBSCAN(eps=eps, min_samples=min(2, len(vectors)), metric=self.metric)
        labels = db.fit_predict(vectors)
        probs = np.ones(len(vectors), dtype=np.float32)
        probs[labels == -1] = 0.0
        return labels, probs, "dbscan"
