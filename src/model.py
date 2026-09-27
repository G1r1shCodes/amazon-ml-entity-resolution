"""Machine Learning Model & Matching Module.

Assigned to: Person 2 (model training, pairwise classifier & decision thresholding)

Handles:
- Training pair generation (positive from GT, hard negatives from blocking candidates)
- Binary match classifier (RandomForest / LightGBM)
- F0.5-optimized threshold tuning
- Batch inference on candidate pairs
"""

import time
import pickle
from typing import Any, Dict, List, Set, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve

from src.features import FEATURE_NAMES, extract_batch_features


class EntityMatchingModel:
    """Pairwise entity matching classifier with F0.5-optimized thresholding."""

    def __init__(self, threshold: float = 0.5, model_type: str = "lgb"):
        self.threshold = threshold
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
        elif model_type in ("lgb", "lightgbm"):
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=300, max_depth=8, learning_rate=0.05,
                    num_leaves=31, scale_pos_weight=5.0, random_state=42, n_jobs=-1,
                    verbose=-1,
                )
            except ImportError:
                print("  ⚠️ LightGBM not installed — falling back to GradientBoostingClassifier")
                self.model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )
        elif model_type in ("xgb", "xgboost"):
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    scale_pos_weight=5.0, random_state=42, n_jobs=-1,
                    eval_metric="logloss",
                )
            except ImportError:
                print("  ⚠️ XGBoost not installed — falling back to GradientBoostingClassifier")
                self.model = GradientBoostingClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )
        elif model_type == "gbdt":
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """Train binary matching classifier.

        Returns dict with training metrics.
        """
        t0 = time.time()
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        print(f"Training {self.model_type} on {len(X_train):,} samples "
              f"({n_pos:,} pos, {n_neg:,} neg, ratio 1:{n_neg//max(n_pos,1)})...",
              flush=True)

        self.model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Trained in {train_time:.1f}s", flush=True)

        # Feature importances
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            print("  Feature importances:")
            for rank, idx in enumerate(sorted_idx[:5]):
                print(f"    {rank+1}. {FEATURE_NAMES[idx]}: {importances[idx]:.4f}")

        return {"train_time": train_time, "n_train": len(X_train)}

    def tune_threshold_f05(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Find threshold that maximizes F0.5 on validation set.

        F0.5 = (1.25 * P * R) / (0.25 * P + R), emphasizes precision.
        """
        probs = self.predict_probabilities(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

        # Compute F0.5 for each threshold
        f05_scores = np.where(
            (0.25 * precisions[:-1] + recalls[:-1]) > 0,
            (1.25 * precisions[:-1] * recalls[:-1]) / (0.25 * precisions[:-1] + recalls[:-1]),
            0.0,
        )

        best_idx = np.argmax(f05_scores)
        best_threshold = thresholds[best_idx]
        best_f05 = f05_scores[best_idx]
        best_p = precisions[best_idx]
        best_r = recalls[best_idx]

        self.threshold = float(best_threshold)
        print(f"  Optimal threshold: {self.threshold:.4f} "
              f"(F0.5={best_f05:.4f}, P={best_p:.4f}, R={best_r:.4f})", flush=True)
        return self.threshold

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict match probabilities for candidate pairs."""
        return self.model.predict_proba(X)[:, 1]

    def predict_matches(self, X: np.ndarray) -> np.ndarray:
        """Predict binary matches based on decision threshold."""
        probs = self.predict_probabilities(X)
        return (probs >= self.threshold).astype(int)

    def save(self, path: str) -> None:
        """Save model + threshold to disk."""
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "threshold": self.threshold,
                         "model_type": self.model_type}, f)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EntityMatchingModel":
        """Load model + threshold from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(threshold=data["threshold"], model_type=data["model_type"])
        obj.model = data["model"]
        return obj


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    n_folds: int = 3,
    model_type: str = "rf",
) -> Tuple[EntityMatchingModel, Dict[str, float]]:
    """Train model with cross-validation and tune threshold.

    Uses stratified K-fold to ensure positive/negative balance in each fold.
    Tunes threshold on the last fold's validation set.

    Returns the final trained model and evaluation metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type} with {n_folds}-fold CV")
    print(f"{'='*60}")

    # Filter out unlabeled pairs (y == -1)
    mask = y >= 0
    X_labeled = X[mask]
    y_labeled = y[mask]

    if len(X_labeled) == 0:
        raise ValueError("No labeled pairs found. Provide gt_map to extract_batch_features().")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        print(f"\n--- Fold {fold+1}/{n_folds} ---")
        X_train, X_val = X_labeled[train_idx], X_labeled[val_idx]
        y_train, y_val = y_labeled[train_idx], y_labeled[val_idx]

        model = EntityMatchingModel(model_type=model_type)
        model.train(X_train, y_train)
        threshold = model.tune_threshold_f05(X_val, y_val)

        # Evaluate on validation
        preds = model.predict_matches(X_val)
        tp = int(((preds == 1) & (y_val == 1)).sum())
        fp = int(((preds == 1) & (y_val == 0)).sum())
        fn = int(((preds == 0) & (y_val == 1)).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f05 = (1.25 * precision * recall) / max(0.25 * precision + recall, 1e-9)

        fold_metrics.append({"precision": precision, "recall": recall, "f05": f05,
                             "threshold": threshold})
        print(f"  Val: P={precision:.4f} R={recall:.4f} F0.5={f05:.4f}")
        best_model = model  # Keep last fold's model

    # Summary
    avg_f05 = np.mean([m["f05"] for m in fold_metrics])
    avg_p = np.mean([m["precision"] for m in fold_metrics])
    avg_r = np.mean([m["recall"] for m in fold_metrics])
    print(f"\n📊 CV Average: P={avg_p:.4f} R={avg_r:.4f} F0.5={avg_f05:.4f}")

    # Retrain on all labeled data with the best threshold
    print("\nRetraining on full labeled data...")
    final_model = EntityMatchingModel(model_type=model_type)
    final_model.train(X_labeled, y_labeled)
    final_model.threshold = float(np.mean([m["threshold"] for m in fold_metrics]))
    print(f"  Final threshold: {final_model.threshold:.4f}")

    return final_model, {
        "avg_precision": avg_p, "avg_recall": avg_r, "avg_f05": avg_f05,
        "final_threshold": final_model.threshold,
    }
