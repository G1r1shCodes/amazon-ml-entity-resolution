"""Data Loading and IO Utilities for Amazon ML Entity Resolution Challenge."""

import os
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd


def load_source_data(file_path: str) -> pd.DataFrame:
    """Load entity records from a tab-separated source TSV file.

    Columns: entity_id, business_name, business_address, country
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    df = pd.read_csv(file_path, sep="\t", dtype=str)
    # Fill NA values with empty strings for text processing safety
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")
    return df


def load_ground_truth(file_path: str) -> Dict[str, Set[str]]:
    """Load training ground truth mappings from train_ground_truth.tsv.

    Returns a dictionary mapping source1_entity_id -> set of matched_entity_ids.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found at: {file_path}")

    df = pd.read_csv(file_path, sep="\t", dtype=str)
    gt_map: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if matched_str.strip():
            matched_set = set(m.strip() for m in matched_str.split(",") if m.strip())
        else:
            matched_set = set()
        gt_map[s1_id] = matched_set

    return gt_map


def save_results(df_results: pd.DataFrame, output_path: str, col_name: str = "matched_entity_ids") -> None:
    """Save formatted predictions/candidates to a TSV output file.

    Columns: source1_entity_id, matched_entity_ids (or candidate_entity_ids)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results = df_results[["source1_entity_id", col_name]].copy()
    df_results[col_name] = df_results[col_name].fillna("")
    df_results.to_csv(output_path, sep="\t", index=False)
    print(f"Successfully saved {len(df_results)} rows to {output_path}")
