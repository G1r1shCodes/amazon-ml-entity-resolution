"""Machine Learning Model & Matching Module.

Assigned to: Person 2 (model training, pairwise classifier & decision thresholding)
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class EntityMatchingModel:
    """Classifier wrapper for pairwise entity resolution matching."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train binary matching classifier."""
        print(f"Training pairwise entity matching model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities for candidate pairs."""
        return self.model.predict_proba(X)[:, 1]

    def predict_matches(self, X: np.ndarray) -> np.ndarray:
        """Predict binary matches based on decision threshold."""
        probs = self.predict_probabilities(X)
        return (probs >= self.threshold).astype(int)
