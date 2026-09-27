"""Unit tests for src/evaluate.py."""

import pytest
from src.evaluate import compute_entity_f05, evaluate_predictions, compute_candidate_recall


def test_singleton_correct():
    """Correctly identified singleton should score 1.0 on all metrics."""
    p, r, f05 = compute_entity_f05(set(), set())
    assert p == 1.0
    assert r == 1.0
    assert f05 == 1.0


def test_singleton_false_positive():
    """Predicting matches for a true singleton should score 0.0."""
    p, r, f05 = compute_entity_f05({"S2-123"}, set())
    assert p == 0.0
    assert r == 0.0
    assert f05 == 0.0


def test_false_negative_missed_all():
    """Predicting no matches when there were real matches should score 0.0."""
    p, r, f05 = compute_entity_f05(set(), {"S2-123", "S3-456"})
    assert p == 0.0
    assert r == 0.0
    assert f05 == 0.0


def test_official_readme_example():
    """Verify exact score matching the official challenge README example.

    Pred: [S2-00047, S2-00193, S3-00812]
    True: [S2-00047, S3-00812]
    Expected: Precision = 2/3, Recall = 1.0, F0.5 = 0.7143
    """
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    p, r, f05 = compute_entity_f05(pred, true)

    assert abs(p - (2.0 / 3.0)) < 1e-4
    assert abs(r - 1.0) < 1e-4
    assert abs(f05 - 0.7143) < 1e-3


def test_evaluate_predictions_macro():
    """Test macro-averaging over multiple entities."""
    preds = {
        "S1-1": {"S2-1"},       # True match: 1.0
        "S1-2": set(),           # True singleton: 1.0
        "S1-3": {"S2-99"},      # False positive on singleton: 0.0
    }
    gt = {
        "S1-1": {"S2-1"},
        "S1-2": set(),
        "S1-3": set(),
    }
    res = evaluate_predictions(preds, gt)
    # Average F0.5 = (1.0 + 1.0 + 0.0) / 3 = 0.6667
    assert abs(res["macro_f05"] - (2.0 / 3.0)) < 1e-4
    assert res["singletons_total"] == 2
    assert res["singletons_correct"] == 1


def test_compute_candidate_recall():
    """Test blocking ceiling candidate recall."""
    gt = {
        "S1-1": {"S2-1", "S3-1"}, # 2 matches
        "S1-2": {"S2-2"},         # 1 match
        "S1-3": set(),            # singleton (ignored in recall calculation)
    }
    cands = {
        "S1-1": {"S2-1", "S2-99"}, # captured 1 of 2
        "S1-2": {"S2-2"},         # captured 1 of 1
        "S1-3": {"S3-55"},
    }
    res = compute_candidate_recall(cands, gt)
    # Total true matches = 3. Captured = 2 (S2-1 and S2-2). Recall = 2/3 = 0.6667
    assert abs(res["candidate_recall"] - (2.0 / 3.0)) < 1e-4
    assert res["captured_true_matches"] == 2
    assert res["total_true_matches"] == 3
