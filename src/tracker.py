"""Experiment Tracking Module for Amazon ML Entity Resolution.

Role: Person 3 (Evaluation, Validation & Experiment Tracking)

Maintains a lightweight, persistent log of model runs, thresholds,
candidate recall, and Macro F0.5 scores in experiments/experiments.csv.
"""

import os
from datetime import datetime
from typing import Optional
import pandas as pd

EXPERIMENT_CSV_PATH = "experiments/experiments.csv"
EXPERIMENT_COLUMNS = [
    "experiment_id",
    "timestamp",
    "blocking_version",
    "feature_version",
    "model",
    "threshold",
    "macro_precision",
    "macro_recall",
    "macro_f05",
    "candidate_recall",
    "notes",
]


def log_experiment(
    experiment_id: str,
    blocking_version: str,
    feature_version: str,
    model: str,
    threshold: float,
    macro_precision: float,
    macro_recall: float,
    macro_f05: float,
    candidate_recall: Optional[float] = None,
    notes: str = "",
    csv_path: str = EXPERIMENT_CSV_PATH,
) -> None:
    """Record an experiment result to the tracking CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "blocking_version": blocking_version,
        "feature_version": feature_version,
        "model": model,
        "threshold": round(threshold, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f05": round(macro_f05, 4),
        "candidate_recall": round(candidate_recall, 4) if candidate_recall is not None else "",
        "notes": notes,
    }

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record], columns=EXPERIMENT_COLUMNS)

    df.to_csv(csv_path, index=False)
    print(f"Logged experiment '{experiment_id}' to {csv_path} (Macro F0.5: {macro_f05:.4f})")


def print_experiment_leaderboard(csv_path: str = EXPERIMENT_CSV_PATH) -> None:
    """Print all logged experiments sorted by Macro F0.5 descending."""
    if not os.path.exists(csv_path):
        print(f"No experiments logged yet at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"Experiment log {csv_path} is empty.")
        return

    df_sorted = df.sort_values(by="macro_f05", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("                     INTERNAL EXPERIMENT LEADERBOARD                         ")
    print("=" * 90)
    print(
        f"{'Exp ID':<10} | {'Model':<15} | {'Thresh':<8} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Cand Recall':<12}"
    )
    print("-" * 90)
    for idx, row in df_sorted.iterrows():
        cand_r = f"{row['candidate_recall']:.4f}" if pd.notna(row["candidate_recall"]) and row["candidate_recall"] != "" else "N/A"
        is_best = " * (TOP)" if idx == 0 else ""
        print(
            f"{str(row['experiment_id']):<10} | {str(row['model']):<15} | {float(row['threshold']):<8.2f} | {float(row['macro_f05']):<12.4f} | {float(row['macro_precision']):<10.4f} | {float(row['macro_recall']):<10.4f} | {cand_r:<12}{is_best}"
        )
    print("=" * 90 + "\n")
