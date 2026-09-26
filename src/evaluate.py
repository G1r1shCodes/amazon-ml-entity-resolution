"""Evaluation Module: Macro F0.5 Metric Computation.

Assigned to: Person 3 (evaluation, validation & pipeline tracking)
"""

from typing import Dict, List, Set, Tuple
import pandas as pd


def compute_entity_f05(pred_set: Set[str], true_set: Set[str]) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F0.5 score for a single Source 1 entity.

    Handles singletons (0-match entities) according to challenge rules:
    - Empty True & Empty Pred => Precision=1.0, Recall=1.0, F0.5=1.0
    - Empty True & Non-Empty Pred => Precision=0.0, Recall=0.0, F0.5=0.0
    - Non-Empty True & Empty Pred => Precision=0.0, Recall=0.0, F0.5=0.0
    """
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not true_set or not pred_set:
        return 0.0, 0.0, 0.0

    intersection = len(pred_set.intersection(true_set))
    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)

    if (0.25 * precision + recall) == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall)

    return precision, recall, f05


def evaluate_predictions(
    predictions_map: Dict[str, Set[str]], ground_truth_map: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Calculate macro-averaged Precision, Recall, and F0.5 score across all Source 1 entities."""
    total_entities = len(ground_truth_map)
    if total_entities == 0:
        return {"precision": 0.0, "recall": 0.0, "f05_score": 0.0}

    total_p, total_r, total_f05 = 0.0, 0.0, 0.0

    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        p, r, f05 = compute_entity_f05(pred_set, true_set)
        total_p += p
        total_r += r
        total_f05 += f05

    macro_precision = total_p / total_entities
    macro_recall = total_r / total_entities
    macro_f05 = total_f05 / total_entities

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f05": macro_f05,
    }
