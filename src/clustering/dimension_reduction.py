"""
Dimensionality Reduction & Manifold Projection.
Projects high-dimensional artwork descriptors to 2D/3D manifolds via UMAP, t-SNE, or PCA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Check for umap-learn availability
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


@dataclass
class ManifoldProjection:
    """Projected 2D or 3D coordinates and explained variance."""
    coordinates: np.ndarray  # (N, n_components)
    method_used: str
    n_components: int
    variance_explained: Optional[float] = None


class DimensionReducer:
    """Reduces artwork descriptors to low-dimensional visualization and clustering spaces."""

    def __init__(
        self,
        n_components: int = 2,
        method: str = "umap",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.method = method.lower()
        self.random_state = random_state
        self.fitted_model = None

    def fit_transform(self, vectors: np.ndarray) -> ManifoldProjection:
        """
        Projects high-dimensional descriptors to low-dimensional manifold.
        """
        arr = np.asarray(vectors, dtype=np.float32)
        n_samples, n_features = arr.shape

        if n_samples < self.n_components:
            raise ValueError(
                f"Sample count ({n_samples}) must be >= n_components ({self.n_components})"
            )

        # Attempt UMAP
        if self.method == "umap" and HAS_UMAP and n_samples >= 4:
            try:
                n_neighbors = min(15, n_samples - 1)
                reducer = umap.UMAP(
                    n_components=self.n_components,
                    n_neighbors=n_neighbors,
                    min_dist=0.1,
                    metric="cosine",
                    random_state=self.random_state,
                )
                coords = reducer.fit_transform(arr)
                self.fitted_model = reducer
                return ManifoldProjection(
                    coordinates=coords.astype(np.float32),
                    method_used="umap",
                    n_components=self.n_components,
                )
            except Exception as e:
                logger.warning("UMAP failed (%s), falling back to PCA.", e)

        # Fallback to PCA
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        coords = pca.fit_transform(arr)
        self.fitted_model = pca
        var_expl = float(np.sum(pca.explained_variance_ratio_))

        return ManifoldProjection(
            coordinates=coords.astype(np.float32),
            method_used="pca",
            n_components=self.n_components,
            variance_explained=var_expl,
        )
