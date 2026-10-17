"""
Artist Reference Profiling Engine.
Constructs statistical manifolds (multivariate Gaussian / Mahalanobis distance)
from an artist's consensus canonical corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class DistanceMetric(str, Enum):
    MAHALANOBIS = "mahalanobis"
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


@dataclass
class ArtistReferenceProfile:
    """Statistical profile capturing the invariant facture distribution of an artist."""
    artist_name: str
    centroid: np.ndarray  # Mean vector (D,)
    covariance: np.ndarray  # Covariance matrix (D, D)
    inv_covariance: np.ndarray  # Regularized inverse covariance (D, D)
    num_reference_artworks: int
    num_reference_patches: int
    median_distance: float
    distance_std: float
    outlier_threshold_p95: float
    outlier_threshold_p99: float
    patch_dim: int

    @classmethod
    def fit_from_patch_vectors(
        cls,
        artist_name: str,
        patch_vectors: np.ndarray,
        num_artworks: int = 1,
        regularization: float = 1e-4,
    ) -> ArtistReferenceProfile:
        """
        Fits a multivariate reference distribution from patch vectors (N, D).
        """
        if len(patch_vectors) < 2:
            raise ValueError("Need at least 2 patches to compute artist distribution")

        vectors = np.asarray(patch_vectors, dtype=np.float64)
        n, d = vectors.shape

        # Normalize rows to unit sphere if not already normalized
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

        centroid = np.mean(vectors, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid /= c_norm

        # Empirical covariance with Ledoit-Wolf-like ridge regularization
        cov = np.cov(vectors, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])

        # Add Tikhonov / Ridge regularization to guarantee invertibility
        cov_reg = cov + np.eye(d) * (regularization + 1e-5 * float(np.trace(cov)) / d)

        try:
            inv_cov = np.linalg.pinv(cov_reg)
        except np.linalg.LinAlgError:
            inv_cov = np.eye(d)

        # Compute empirical distance distribution for calibration
        diff = vectors - centroid
        left = np.dot(diff, inv_cov)
        maha_sq = np.sum(left * diff, axis=1)
        distances = np.sqrt(np.maximum(0.0, maha_sq))

        med_dist = float(np.median(distances))
        std_dist = float(np.std(distances))
        p95 = float(np.percentile(distances, 95))
        p99 = float(np.percentile(distances, 99))

        return cls(
            artist_name=artist_name,
            centroid=centroid.astype(np.float32),
            covariance=cov.astype(np.float32),
            inv_covariance=inv_cov.astype(np.float32),
            num_reference_artworks=num_artworks,
            num_reference_patches=n,
            median_distance=med_dist,
            distance_std=std_dist,
            outlier_threshold_p95=p95,
            outlier_threshold_p99=p99,
            patch_dim=d,
        )

    def compute_distance(
        self,
        vector: np.ndarray,
        metric: DistanceMetric = DistanceMetric.MAHALANOBIS,
    ) -> float:
        """
        Computes distance between an individual patch or artwork vector and the artist profile.
        """
        vec = np.asarray(vector, dtype=np.float64).ravel()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        if metric == DistanceMetric.COSINE:
            cos_sim = float(np.dot(vec, self.centroid))
            return max(0.0, 1.0 - cos_sim)

        if metric == DistanceMetric.EUCLIDEAN:
            return float(np.linalg.norm(vec - self.centroid))

        # Default: Mahalanobis distance
        diff = vec - self.centroid
        dist_sq = float(np.dot(np.dot(diff, self.inv_covariance), diff))
        return float(np.sqrt(max(0.0, dist_sq)))

    def compute_attribution_confidence(
        self,
        vector: np.ndarray,
        metric: DistanceMetric = DistanceMetric.MAHALANOBIS,
    ) -> float:
        """
        Converts distance into a calibrated confidence score [0.0, 1.0].
        1.0 = exactly matches master distribution median.
        0.0 = extreme outlier (beyond p99 threshold).
        """
        dist = self.compute_distance(vector, metric)

        if metric == DistanceMetric.COSINE:
            # Cosine distance ranges [0, 2]
            return float(np.clip(1.0 - dist, 0.0, 1.0))

        # Exponential decay calibrated by median and p95 thresholds
        scale = max(1e-4, self.outlier_threshold_p95 - self.median_distance)
        z = (dist - self.median_distance) / scale
        z_clipped = np.clip(z * 1.5, -50.0, 50.0)
        conf = 1.0 / (1.0 + np.exp(z_clipped))
        return float(np.clip(conf, 0.0, 1.0))
