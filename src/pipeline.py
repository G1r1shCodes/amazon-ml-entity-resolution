"""End-to-End Pipeline Execution Script.

Assigned to: Person 3 (pipeline orchestration & submission generation)
Executes end-to-end entity resolution flow:
Data Loading -> Blocking (Candidates) -> Feature Extraction -> Model Inference -> Validation -> Output Export
"""

import os
import sys
import pandas as pd
from src.blocking import generate_candidate_pairs
from src.data import load_ground_truth, load_source_data, save_results
from src.evaluate import evaluate_predictions


def run_pipeline(data_dir: str = "dataset/test", output_dir: str = "output") -> None:
    """Run entity resolution pipeline for test data and export output files."""
    print("=== Amazon ML Entity Resolution Pipeline ===")

    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s2_path = os.path.join(data_dir, "test_source2.tsv")
    s3_path = os.path.join(data_dir, "test_source3.tsv")

    print(f"Loading data from {data_dir}...")
    df_s1 = load_source_data(s1_path)
    df_s2 = load_source_data(s2_path)
    df_s3 = load_source_data(s3_path)

    print(f"Source 1 records: {len(df_s1)}")
    print(f"Source 2 records: {len(df_s2)}")
    print(f"Source 3 records: {len(df_s3)}")

    # 1. Blocking / Candidate Pair Generation
    print("\n--- Phase 1: Blocking & Candidate Generation ---")
    df_candidates = generate_candidate_pairs(df_s1, df_s2, df_s3)
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    save_results(df_candidates, candidate_file, col_name="candidate_entity_ids")

    # 2. Matching Model Inference (Stub / Baseline)
    print("\n--- Phase 2: Matching Model Prediction ---")
    df_matches = df_candidates.rename(columns={"candidate_entity_ids": "matched_entity_ids"}).copy()
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    save_results(df_matches, matching_file, col_name="matched_entity_ids")

    print("\nPipeline execution finished successfully.")


if __name__ == "__main__":
    run_pipeline()
