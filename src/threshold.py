"""Threshold Tuning and Decision Analysis Module.

Role: Person 3 (Evaluation, Validation & Pipeline Orchestration)

Sweeps decision thresholds on pairwise match probabilities to find the
threshold that maximizes the official Macro F0.5 score on validation data.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from src.evaluate import compute_entity_f05, evaluate_predictions


def sweep_thresholds(
    df_scores: pd.DataFrame,
    ground_truth_map: Dict[str, Set[str]],
    all_s1_ids: Set[str],
    thresholds: Optional[List[float]] = None,
) -> pd.DataFrame:
    """Sweep a list of thresholds across pairwise match scores.

    Args:
        df_scores: DataFrame with ['source1_entity_id', 'candidate_entity_id', 'score'].
        ground_truth_map: Dict mapping s1_id -> set of actual matched IDs.
        all_s1_ids: Complete set of all Source 1 entity IDs in the evaluation set.
        thresholds: List of thresholds to test. Defaults to 0.10 through 0.95.

    Returns:
        DataFrame containing metrics for each evaluated threshold.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]

    print(f"Loaded {len(df_scores):,} scored candidate pairs.")
    print(f"Total Source 1 entities in evaluation set: {len(all_s1_ids):,}")

    records = []

    for thresh in thresholds:
        # Filter candidate pairs that meet or exceed the threshold
        df_pos = df_scores[df_scores["score"] >= thresh]

        # Group matched candidate IDs by source1_entity_id
        pred_map: Dict[str, Set[str]] = {s1: set() for s1 in all_s1_ids}
        grouped = df_pos.groupby("source1_entity_id")["candidate_entity_id"].apply(set)
        for s1, cands in grouped.items():
            if s1 in pred_map:
                pred_map[s1] = cands

        # Evaluate against ground truth using official Macro F0.5
        eval_res = evaluate_predictions(pred_map, ground_truth_map)

        records.append({
            "threshold": thresh,
            "macro_f05": eval_res["macro_f05"],
            "macro_precision": eval_res["macro_precision"],
            "macro_recall": eval_res["macro_recall"],
            "pred_matches": eval_res["total_pred_matches"],
            "true_positives": eval_res["total_true_positives"],
            "singletons_correct": eval_res["singletons_correct"],
            "singletons_accuracy": eval_res["singletons_accuracy"],
        })

    df_results = pd.DataFrame(records)
    return df_results


def print_threshold_table(df_results: pd.DataFrame, best_idx: int) -> None:
    """Print a clean comparison table highlighting the best threshold."""
    print("\n" + "=" * 80)
    print("             THRESHOLD TUNING & OPTIMIZATION REPORT (MACRO F0.5)               ")
    print("=" * 80)
    print(
        f"{'Threshold':<11} | {'Macro F0.5':<12} | {'Precision':<11} | {'Recall':<10} | {'Pred Matches':<14} | {'TP':<8} | {'Singletons Acc':<14}"
    )
    print("-" * 80)

    for idx, row in df_results.iterrows():
        is_best = " * (BEST)" if idx == best_idx else ""
        print(
            f"{row['threshold']:<11.2f} | {row['macro_f05']:<12.4f} | {row['macro_precision']:<11.4f} | {row['macro_recall']:<10.4f} | {int(row['pred_matches']):<14,d} | {int(row['true_positives']):<8,d} | {row['singletons_accuracy']*100:<13.1f}%{is_best}"
        )
    print("=" * 80 + "\n")


def generate_predictions_at_threshold(
    df_scores: pd.DataFrame,
    all_s1_ids: Set[str],
    threshold: float,
    output_path: str,
) -> None:
    """Save matching_results.tsv using the selected threshold."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pos = df_scores[df_scores["score"] >= threshold]

    grouped = df_pos.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: ",".join(sorted(ids)))

    result_rows = []
    for s1 in sorted(all_s1_ids):
        result_rows.append({
            "source1_entity_id": s1,
            "matched_entity_ids": grouped.get(s1, ""),
        })

    df_out = pd.DataFrame(result_rows)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Exported final matching results at threshold {threshold:.2f} to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Find optimal classification threshold for Macro F0.5.")
    parser.add_argument("--scores", "-s", required=True, help="TSV/CSV file containing [source1_entity_id, candidate_entity_id, score]")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth TSV")
    parser.add_argument("--source1", "-s1", required=True, help="Path to Source 1 TSV to guarantee all entities are represented")
    parser.add_argument("--output", "-o", default=None, help="Optional path to export matching_results.tsv at best threshold")
    parser.add_argument("--report", "-r", default=None, help="Optional path to save threshold comparison CSV")
    args = parser.parse_args()

    # Load scores
    print(f"Reading pair prediction scores from: {args.scores}")
    sep = "\t" if args.scores.endswith(".tsv") else ","
    df_scores = pd.read_csv(args.scores, sep=sep)
    for col in ["source1_entity_id", "candidate_entity_id", "score"]:
        if col not in df_scores.columns:
            raise ValueError(f"Missing required column '{col}' in {args.scores}")

    # Load ground truth
    from src.evaluate import load_mapping_from_tsv
    print(f"Reading ground truth from: {args.ground_truth}")
    gt_map = load_mapping_from_tsv(args.ground_truth, "source1_entity_id", "matched_entity_ids")

    # Load Source 1 IDs
    print(f"Reading Source 1 IDs from: {args.source1}")
    df_s1 = pd.read_csv(args.source1, sep="\t", dtype=str, usecols=["entity_id"])
    all_s1_ids = set(df_s1["entity_id"])

    # Sweep thresholds
    df_results = sweep_thresholds(df_scores, gt_map, all_s1_ids)

    best_idx = int(df_results["macro_f05"].idxmax())
    best_thresh = df_results.loc[best_idx, "threshold"]
    best_f05 = df_results.loc[best_idx, "macro_f05"]

    print_threshold_table(df_results, best_idx)

    print(f"Optimal Threshold: {best_thresh:.2f} yielding Macro F0.5: {best_f05:.4f}")

    if args.report:
        df_results.to_csv(args.report, index=False)
        print(f"Threshold report saved to: {args.report}")

    if args.output:
        generate_predictions_at_threshold(df_scores, all_s1_ids, best_thresh, args.output)


if __name__ == "__main__":
    main()
