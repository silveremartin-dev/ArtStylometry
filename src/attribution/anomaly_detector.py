"""
Stylistic Anomaly & Outlier Detector.
Employs Isolation Forests and One-Class SVMs to detect statistical departures
from canonical artist workshop distributions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Detailed anomaly audit report for an artwork or set of patches."""
    is_anomaly: bool
    anomaly_score: float  # Negative = more anomalous, positive = inlier
    confidence_percentile: float
    method_used: str
    num_samples_evaluated: int


class StylisticAnomalyDetector:
    """Unsupervised anomaly detector for fine art facture vectors."""

    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:
        """
        Args:
            method: 'isolation_forest' or 'one_class_svm'.
            contamination: Expected proportion of outliers in training data.
            random_state: Seed for reproducible isolation forest splits.
        """
        self.method = method
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, reference_vectors: np.ndarray) -> StylisticAnomalyDetector:
        """Fits anomaly detector on uncontested canonical artwork descriptors."""
        vectors = np.asarray(reference_vectors, dtype=np.float32)
        if len(vectors) < 2:
            raise ValueError("Need at least 2 samples to fit anomaly detector")

        if self.method == "one_class_svm":
            self.model = OneClassSVM(nu=self.contamination, kernel="rbf", gamma="scale")
        else:
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=100,
            )

        self.model.fit(vectors)
        return self

    def predict(self, test_vector: np.ndarray) -> AnomalyReport:
        """Evaluates a test vector against the learned decision boundary."""
        if self.model is None:
            raise RuntimeError("Anomaly detector must be fitted before predict()")

        vec = np.asarray(test_vector, dtype=np.float32).reshape(1, -1)
        pred = int(self.model.predict(vec)[0])  # +1 inlier, -1 outlier
        score = float(self.model.decision_function(vec)[0])

        is_anomaly = pred == -1

        # Transform score into roughly calibrated percentile [0, 100]
        # In isolation forest, decision_function > 0 is inlier, < 0 is anomaly
        percentile = float(1.0 / (1.0 + np.exp(-score * 5.0)) * 100.0)

        return AnomalyReport(
            is_anomaly=is_anomaly,
            anomaly_score=score,
            confidence_percentile=percentile,
            method_used=self.method,
            num_samples_evaluated=1,
        )
