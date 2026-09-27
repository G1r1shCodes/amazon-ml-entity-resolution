"""Evaluation Module for Amazon ML Entity Resolution Challenge.

Role: Person 3 (Evaluation, Validation & Pipeline Orchestration)

Calculates the official Macro-averaged F0.5 score, Precision, Recall,
Candidate Recall (Blocking Ceiling), and Singleton Metrics.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


def compute_entity_f05(pred_set: Set[str], true_set: Set[str]) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F0.5 score for a single Source 1 entity.

    Follows the official challenge scoring rules:
    - Empty True & Empty Pred => Precision=1.0, Recall=1.0, F0.5=1.0 (Correctly identified singleton)
    - Empty True & Non-Empty Pred => Precision=0.0, Recall=0.0, F0.5=0.0 (False merge on singleton)
    - Non-Empty True & Empty Pred => Precision=0.0, Recall=0.0, F0.5=0.0 (Missed all matches)
    - Non-Empty True & Non-Empty Pred:
        P = |Intersection| / |Pred|
        R = |Intersection| / |True|
        F0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not true_set or not pred_set:
        return 0.0, 0.0, 0.0

    intersection = len(pred_set.intersection(true_set))
    if intersection == 0:
        return 0.0, 0.0, 0.0

    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)

    denom = (0.25 * precision) + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return precision, recall, f05


def parse_id_list(value: any) -> Set[str]:
    """Parse comma-separated entity IDs into a clean set."""
    if pd.isna(value) or not str(value).strip():
        return set()
    return {item.strip() for item in str(value).split(",") if item.strip()}


def load_mapping_from_tsv(file_path: str, id_col: str, list_col: str) -> Dict[str, Set[str]]:
    """Read a TSV file into a dictionary of {s1_id: set_of_ids}."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    mapping: Dict[str, Set[str]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return mapping
        headers = [h.strip() for h in header_line.rstrip("\n").split("\t")]
        if id_col not in headers or list_col not in headers:
            raise ValueError(
                f"File {file_path} missing expected columns '{id_col}' or '{list_col}'. Found: {headers}"
            )
        id_idx = headers.index(id_col)
        list_idx = headers.index(list_col)

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > id_idx:
                s1_id = parts[id_idx].strip()
                list_str = parts[list_idx].strip() if len(parts) > list_idx else ""
                mapping[s1_id] = {x.strip() for x in list_str.split(",") if x.strip()} if list_str else set()
    return mapping


def evaluate_predictions(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    country_map: Optional[Dict[str, str]] = None,
) -> Dict[str, any]:
    """Calculate macro-averaged Precision, Recall, and F0.5 score across all Source 1 entities."""
    total_entities = len(ground_truth_map)
    if total_entities == 0:
        return {
            "total_entities": 0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f05": 0.0,
            "singletons_total": 0,
            "singletons_correct": 0,
        }

    sum_p, sum_r, sum_f05 = 0.0, 0.0, 0.0
    true_singletons, correct_singletons = 0, 0
    total_true_matches, total_pred_matches, total_tp = 0, 0, 0

    country_stats: Dict[str, Dict[str, float]] = {}

    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())

        p, r, f05 = compute_entity_f05(pred_set, true_set)
        sum_p += p
        sum_r += r
        sum_f05 += f05

        tp_count = len(pred_set.intersection(true_set))
        total_tp += tp_count
        total_pred_matches += len(pred_set)
        total_true_matches += len(true_set)

        if not true_set:
            true_singletons += 1
            if not pred_set:
                correct_singletons += 1

        if country_map and s1_id in country_map:
            c = country_map[s1_id]
            if c not in country_stats:
                country_stats[c] = {"sum_p": 0.0, "sum_r": 0.0, "sum_f05": 0.0, "count": 0}
            country_stats[c]["sum_p"] += p
            country_stats[c]["sum_r"] += r
            country_stats[c]["sum_f05"] += f05
            country_stats[c]["count"] += 1

    macro_precision = sum_p / total_entities
    macro_recall = sum_r / total_entities
    macro_f05 = sum_f05 / total_entities

    results = {
        "total_entities": total_entities,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f05": macro_f05,
        "total_true_matches": total_true_matches,
        "total_pred_matches": total_pred_matches,
        "total_true_positives": total_tp,
        "micro_precision": total_tp / total_pred_matches if total_pred_matches > 0 else 0.0,
        "micro_recall": total_tp / total_true_matches if total_true_matches > 0 else 0.0,
        "singletons_total": true_singletons,
        "singletons_correct": correct_singletons,
        "singletons_accuracy": correct_singletons / true_singletons if true_singletons > 0 else 1.0,
    }

    if country_stats:
        results["countries"] = {
            c: {
                "macro_f05": data["sum_f05"] / data["count"],
                "macro_precision": data["sum_p"] / data["count"],
                "macro_recall": data["sum_r"] / data["count"],
                "count": data["count"],
            }
            for c, data in country_stats.items()
        }

    return results


def compute_candidate_recall(
    candidates_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
) -> Dict[str, float]:
    """Compute Candidate Recall (Blocking Ceiling).

    Tells us what percentage of true matches were captured by Person 1's blocking stage.
    If a true match is not in candidates, Person 2's model cannot possibly find it.
    """
    total_true_matches = 0
    captured_true_matches = 0
    entities_with_all_matches_found = 0
    non_singleton_entities = 0

    for s1_id, true_set in ground_truth_map.items():
        if not true_set:
            continue
        non_singleton_entities += 1
        cand_set = candidates_map.get(s1_id, set())
        captured = len(true_set.intersection(cand_set))
        captured_true_matches += captured
        total_true_matches += len(true_set)
        if captured == len(true_set):
            entities_with_all_matches_found += 1

    candidate_recall = captured_true_matches / total_true_matches if total_true_matches > 0 else 1.0
    full_coverage_rate = (
        entities_with_all_matches_found / non_singleton_entities if non_singleton_entities > 0 else 1.0
    )

    return {
        "candidate_recall": candidate_recall,
        "captured_true_matches": captured_true_matches,
        "total_true_matches": total_true_matches,
        "full_coverage_rate": full_coverage_rate,
    }


def print_evaluation_report(
    eval_results: Dict[str, any],
    blocking_results: Optional[Dict[str, float]] = None,
) -> None:
    """Print an executive evaluation summary formatted with clear ASCII tables."""
    print("\n" + "=" * 65)
    print("        AMAZON ML ENTITY RESOLUTION -- EVALUATION REPORT        ")
    print("=" * 65)
    print(f"Total Evaluated S1 Entities : {eval_results['total_entities']:,}")
    print(f"Total Actual Matches        : {eval_results['total_true_matches']:,}")
    print(f"Total Predicted Matches     : {eval_results['total_pred_matches']:,}")
    print(f"True Positive Matches (TP)  : {eval_results['total_true_positives']:,}")
    print("-" * 65)
    print(f"* OFFICIAL LEADERBOARD METRIC *")
    print(f"  MACRO F0.5 SCORE          : {eval_results['macro_f05']:.4f}")
    print(f"  Macro Precision           : {eval_results['macro_precision']:.4f}")
    print(f"  Macro Recall              : {eval_results['macro_recall']:.4f}")
    print("-" * 65)
    print(f"Singletons (0-match businesses):")
    print(
        f"  Total Singletons: {eval_results['singletons_total']:,} | Correctly Identified: {eval_results['singletons_correct']:,} ({eval_results['singletons_accuracy']*100:.1f}%)"
    )

    if blocking_results:
        print("-" * 65)
        print(f"* PERSON 1 BLOCKING QUALITY (RECALL CEILING) *")
        print(f"  Candidate Recall          : {blocking_results['candidate_recall']:.4f} ({blocking_results['captured_true_matches']:,}/{blocking_results['total_true_matches']:,})")
        print(f"  Full Coverage Rate        : {blocking_results['full_coverage_rate']:.4f}")
        if blocking_results["candidate_recall"] < 0.90:
            print("  [WARNING]: Blocking Recall < 90%. Person 1's blocking is discarding true matches!")

    if "countries" in eval_results:
        print("-" * 65)
        print("Breakdown By Country:")
        print(f"  {'Country':<10} | {'Entities':<10} | {'Macro F0.5':<12} | {'Macro Precision':<15} | {'Macro Recall':<12}")
        print("  " + "-" * 60)
        for country, c_data in eval_results["countries"].items():
            print(
                f"  {country:<10} | {c_data['count']:<10,d} | {c_data['macro_f05']:<12.4f} | {c_data['macro_precision']:<15.4f} | {c_data['macro_recall']:<12.4f}"
            )

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution Predictions against Ground Truth.")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions TSV (matching_results.tsv)")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth TSV (train_ground_truth.tsv)")
    parser.add_argument("--candidates", "-c", default=None, help="Optional path to candidate_pairs.tsv for blocking recall audit")
    parser.add_argument("--source1", "-s", default=None, help="Optional path to source1.tsv for country breakdown")
    args = parser.parse_args()

    print(f"Loading predictions from: {args.predictions}")
    preds = load_mapping_from_tsv(args.predictions, "source1_entity_id", "matched_entity_ids")

    print(f"Loading ground truth from: {args.ground_truth}")
    gt = load_mapping_from_tsv(args.ground_truth, "source1_entity_id", "matched_entity_ids")

    country_map = None
    if args.source1 and os.path.isfile(args.source1):
        print(f"Loading source 1 metadata for country breakdown: {args.source1}")
        df_s1 = pd.read_csv(args.source1, sep="\t", dtype=str, usecols=["entity_id", "country"])
        country_map = dict(zip(df_s1["entity_id"], df_s1["country"]))

    eval_results = evaluate_predictions(preds, gt, country_map=country_map)

    blocking_results = None
    if args.candidates and os.path.isfile(args.candidates):
        print(f"Loading candidate pairs from: {args.candidates}")
        candidates = load_mapping_from_tsv(args.candidates, "source1_entity_id", "candidate_entity_ids")
        blocking_results = compute_candidate_recall(candidates, gt)

    print_evaluation_report(eval_results, blocking_results)


if __name__ == "__main__":
    main()
