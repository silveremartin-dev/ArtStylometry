"""
Artwork Descriptor & Patch Aggregation Engine.
Aggregates patch-level micro-embeddings into canvas-level representations
via statistical moment pooling and VLAD (Vector of Locally Aggregated Descriptors).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .micro_extractor import PatchEmbeddingResult
from .macro_extractor import CompositionFeatures


@dataclass
class ArtworkDescriptor:
    """Consolidated representation of an entire artwork combining micro and macro cues."""
    artwork_id: str
    feature_vector: np.ndarray  # L2-normalized 1D descriptor
    num_valid_patches: int
    mean_cleanliness: float
    mean_gradient_energy: float
    composition: Optional[CompositionFeatures] = None


class PatchAggregator:
    """Aggregates an ensemble of patch embeddings into a single fingerprint."""

    def __init__(
        self,
        method: str = "statistical",
        num_vlad_centers: int = 8,
        include_macro: bool = True,
    ) -> None:
        """
        Args:
            method: 'statistical' (mean/std/skew) or 'vlad' (Vector of Locally Aggregated Descriptors).
            num_vlad_centers: Number of clusters for VLAD codebook.
            include_macro: If True, appends macro-composition features to the final vector.
        """
        self.method = method
        self.num_vlad_centers = num_vlad_centers
        self.include_macro = include_macro
        self.codebook: Optional[MiniBatchKMeans] = None

    def fit_vlad_codebook(self, all_patch_vectors: np.ndarray) -> None:
        """Fits visual vocabulary centers across a corpus of training patches."""
        if len(all_patch_vectors) < self.num_vlad_centers:
            raise ValueError(
                f"Need at least {self.num_vlad_centers} patches to fit VLAD codebook, got {len(all_patch_vectors)}"
            )
        self.codebook = MiniBatchKMeans(
            n_clusters=self.num_vlad_centers,
            random_state=42,
            batch_size=256,
        )
        self.codebook.fit(all_patch_vectors)

    def aggregate(
        self,
        patch_results: List[PatchEmbeddingResult],
        artwork_id: str = "artwork",
        composition: Optional[CompositionFeatures] = None,
    ) -> ArtworkDescriptor:
        """
        Compiles patch embeddings into an ArtworkDescriptor.
        """
        valid_results = [r for r in patch_results if r.is_valid]
        if not valid_results:
            raise ValueError("Cannot aggregate empty or invalid patch results")

        # Cleanliness weighting: weight patches by their crack-free cleanliness score
        weights = np.array([max(0.1, r.cleanliness) for r in valid_results], dtype=np.float32)
        weights /= np.sum(weights)

        vectors = np.stack([r.vector for r in valid_results], axis=0)  # (N, D)

        if self.method == "vlad" and self.codebook is not None:
            micro_vec = self._aggregate_vlad(vectors, weights)
        else:
            micro_vec = self._aggregate_statistical(vectors, weights)

        # Fuse with macro-composition if available
        if self.include_macro and composition is not None:
            fused = np.concatenate([micro_vec, composition.feature_vector])
        else:
            fused = micro_vec

        norm = np.linalg.norm(fused)
        if norm > 0:
            fused /= norm

        mean_clean = float(np.mean([r.cleanliness for r in valid_results]))
        mean_energy = float(np.mean([r.gradient_energy for r in valid_results]))

        return ArtworkDescriptor(
            artwork_id=artwork_id,
            feature_vector=fused.astype(np.float32),
            num_valid_patches=len(valid_results),
            mean_cleanliness=mean_clean,
            mean_gradient_energy=mean_energy,
            composition=composition,
        )

    def _aggregate_statistical(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Multi-moment weighted statistical pooling:
        - Weighted Mean
        - Weighted Standard Deviation
        - 10th and 90th Percentiles
        """
        # Weighted mean
        mean = np.sum(vectors * weights[:, np.newaxis], axis=0)

        # Weighted standard deviation
        variance = np.sum(weights[:, np.newaxis] * ((vectors - mean) ** 2), axis=0)
        std = np.sqrt(np.maximum(1e-8, variance))

        # Quantile profile
        p10 = np.percentile(vectors, 10, axis=0)
        p90 = np.percentile(vectors, 90, axis=0)

        combined = np.concatenate([mean, std, p10, p90])
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined /= norm
        return combined

    def _aggregate_vlad(self, vectors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Computes VLAD representation with power normalization."""
        centers = self.codebook.cluster_centers_  # (K, D)
        labels = self.codebook.predict(vectors)  # (N,)
        k, d = centers.shape

        vlad = np.zeros((k, d), dtype=np.float32)
        for i, (vec, lbl, w) in enumerate(zip(vectors, labels, weights)):
            residual = (vec - centers[lbl]) * w
            vlad[lbl] += residual

        # Power normalization (signed square root)
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))

        # L2 normalization
        vlad_flat = vlad.flatten()
        norm = np.linalg.norm(vlad_flat)
        if norm > 0:
            vlad_flat /= norm
        return vlad_flat
