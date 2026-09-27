"""End-to-End Pipeline Execution Script.

Role: Person 3 (Pipeline Orchestration, Integration & Output Generation)

Connects Person 1 (Blocking) + Person 2 (Features & Model) + Person 3 (Evaluation & Validation)
Supports both validation mode (--mode val) and competition test inference (--mode test).
"""

import argparse
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Optional, Set, Tuple
import joblib
import numpy as np
import pandas as pd

from src.blocking import generate_blocking_keys
from src.data import load_ground_truth, load_source_data, save_results
from src.evaluate import (
    compute_candidate_recall,
    evaluate_predictions,
    print_evaluation_report,
)
from src.features import compute_pair_features
from src.model import EntityMatchingModel
from src.threshold import generate_predictions_at_threshold, sweep_thresholds
from src.tracker import log_experiment, print_experiment_leaderboard
from src.validate_submission import validate_submission_files


def build_fast_index(df_source: pd.DataFrame) -> Dict[str, List[str]]:
    """Build blocking index using fast tuple unpacking instead of slow iterrows."""
    index: Dict[str, List[str]] = {}
    for eid, name, addr, country in zip(
        df_source["entity_id"],
        df_source["business_name"].fillna(""),
        df_source["business_address"].fillna(""),
        df_source["country"].fillna(""),
    ):
        for key in generate_blocking_keys(name, addr, country):
            if key not in index:
                index[key] = []
            index[key].append(eid)
    return index


def generate_candidate_pairs_fast(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    max_candidates_per_entity: int = 150,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Generate candidate pairs efficiently with capped candidate set per entity to conserve RAM."""
    print("Building blocking index for Source 2 and Source 3...")
    t0 = time.time()
    idx_s2 = build_fast_index(df_s2)
    idx_s3 = build_fast_index(df_s3)
    print(f"Blocking index built in {time.time() - t0:.2f}s ({len(idx_s2):,} S2 keys, {len(idx_s3):,} S3 keys).")

    print(f"Generating candidate matches for {len(df_s1):,} Source 1 entities...")
    t1 = time.time()
    candidate_records = []
    pair_list: List[Tuple[str, str]] = []

    for eid, name, addr, country in zip(
        df_s1["entity_id"],
        df_s1["business_name"].fillna(""),
        df_s1["business_address"].fillna(""),
        df_s1["country"].fillna(""),
    ):
        matched_cands: Set[str] = set()
        for key in generate_blocking_keys(name, addr, country):
            if key in idx_s2:
                matched_cands.update(idx_s2[key])
            if key in idx_s3:
                matched_cands.update(idx_s3[key])

        # Limit max candidates if candidate set is excessively large
        if len(matched_cands) > max_candidates_per_entity:
            cands_list = sorted(list(matched_cands))[:max_candidates_per_entity]
        else:
            cands_list = sorted(list(matched_cands))

        for cid in cands_list:
            pair_list.append((eid, cid))

        candidate_records.append({
            "source1_entity_id": eid,
            "candidate_entity_ids": ",".join(cands_list),
        })

    df_candidates = pd.DataFrame(candidate_records)
    print(f"Candidate generation completed in {time.time() - t1:.2f}s. Total candidate pairs: {len(pair_list):,}")
    return df_candidates, pair_list


def prepare_entity_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Convert dataframe to lookup dict with pre-normalized text to accelerate feature extraction."""
    from src.normalize import normalize_address, normalize_business_name
    records = {}
    for eid, name, addr, country in zip(
        df["entity_id"],
        df["business_name"].fillna(""),
        df["business_address"].fillna(""),
        df["country"].fillna(""),
    ):
        records[eid] = {
            "business_name": name,
            "business_address": addr,
            "country": country,
            "norm_name": normalize_business_name(name),
            "norm_addr": normalize_address(addr),
        }
    return records


def extract_features_for_pairs(
    pair_list: List[Tuple[str, str]],
    s1_dict: Dict[str, Dict[str, str]],
    s23_dict: Dict[str, Dict[str, str]],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute feature vectors for candidate pairs in batches."""
    print(f"Extracting similarity features for {len(pair_list):,} candidate pairs...", flush=True)
    t0 = time.time()

    feature_rows = []
    valid_pairs = []

    for s1_id, cand_id in pair_list:
        rec1 = s1_dict.get(s1_id)
        rec2 = s23_dict.get(cand_id)
        if not rec1 or not rec2:
            continue

        feats = compute_pair_features(rec1, rec2)
        feature_rows.append(feats)
        valid_pairs.append((s1_id, cand_id))

    df_feats = pd.DataFrame(feature_rows)
    X = df_feats.values
    df_pairs = pd.DataFrame(valid_pairs, columns=["source1_entity_id", "candidate_entity_id"])

    print(f"Features extracted in {time.time() - t0:.2f}s across {X.shape[1]} similarity dimensions.", flush=True)
    return df_pairs, X


def run_validation_pipeline(
    data_dir: str = "dataset/val_sample",
    output_dir: str = "outputs/val_run",
    model_save_path: str = "outputs/model.joblib",
) -> None:
    """Execute validation flow: blocking -> feature engineering -> training -> tuning -> evaluation."""
    print("=" * 70)
    print("           AMAZON ML ENTITY RESOLUTION -- VALIDATION PIPELINE           ")
    print("=" * 70)

    # 1. Load Data
    s1_path = os.path.join(data_dir, "val_source1.tsv")
    s2_path = os.path.join(data_dir, "val_source2.tsv")
    s3_path = os.path.join(data_dir, "val_source3.tsv")
    gt_path = os.path.join(data_dir, "val_ground_truth.tsv")

    print(f"Loading validation datasets from {data_dir}...")
    df_s1 = load_source_data(s1_path)
    df_s2 = load_source_data(s2_path)
    df_s3 = load_source_data(s3_path)
    gt_map = load_ground_truth(gt_path)

    # 2. Blocking
    df_candidates, pair_list = generate_candidate_pairs_fast(df_s1, df_s2, df_s3)
    cand_file = os.path.join(output_dir, "candidate_pairs.tsv")
    save_results(df_candidates, cand_file, col_name="candidate_entity_ids")

    # Audit Person 1's blocking
    cands_map = {row["source1_entity_id"]: set(row["candidate_entity_ids"].split(",")) if row["candidate_entity_ids"] else set() for _, row in df_candidates.iterrows()}
    blocking_results = compute_candidate_recall(cands_map, gt_map)

    # 3. Features
    print("Preparing entity lookup tables with normalization caching...", flush=True)
    s1_dict = prepare_entity_lookup(df_s1)
    s23_dict = prepare_entity_lookup(df_s2)
    s23_dict.update(prepare_entity_lookup(df_s3))
    df_pairs, X = extract_features_for_pairs(pair_list, s1_dict, s23_dict)

    # 4. Create Ground Truth Binary Labels (y = 1 if cand_id in gt, else 0)
    y = []
    for s1_id, cand_id in zip(df_pairs["source1_entity_id"], df_pairs["candidate_entity_id"]):
        y.append(1 if cand_id in gt_map.get(s1_id, set()) else 0)
    y = np.array(y)

    print(f"Training pairs distribution: {np.sum(y):,} Positive matches, {len(y) - np.sum(y):,} Negative pairs.")

    # 5. Model Training (Person 2)
    model = EntityMatchingModel()
    model.train(X, y)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(model, model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # 6. Predict Match Probabilities
    scores = model.predict_probabilities(X)
    df_pairs["score"] = scores

    # 7. Threshold Tuning (Person 3)
    all_s1_ids = set(df_s1["entity_id"])
    print("\nSweeping decision thresholds to optimize Macro F0.5...")
    df_thresh = sweep_thresholds(df_pairs, gt_map, all_s1_ids)

    best_idx = int(df_thresh["macro_f05"].idxmax())
    best_thresh = float(df_thresh.loc[best_idx, "threshold"])
    best_f05 = float(df_thresh.loc[best_idx, "macro_f05"])
    best_p = float(df_thresh.loc[best_idx, "macro_precision"])
    best_r = float(df_thresh.loc[best_idx, "macro_recall"])

    # 8. Generate Predictions at Best Threshold
    match_file = os.path.join(output_dir, "matching_results.tsv")
    generate_predictions_at_threshold(df_pairs, all_s1_ids, best_thresh, match_file)

    # 9. Evaluate & Print Report
    country_map = dict(zip(df_s1["entity_id"], df_s1["country"]))
    preds_map = load_ground_truth(match_file) # Re-use loader for s1 -> set
    eval_results = evaluate_predictions(preds_map, gt_map, country_map=country_map)
    print_evaluation_report(eval_results, blocking_results)

    # 10. Log Experiment (Person 3 Tracking)
    log_experiment(
        experiment_id="EXP_VAL_BASELINE",
        blocking_version="blocking_v1_prefix_firstword",
        feature_version="features_v1_jaccard_levenshtein",
        model="RandomForest_depth10",
        threshold=best_thresh,
        macro_precision=best_p,
        macro_recall=best_r,
        macro_f05=best_f05,
        candidate_recall=blocking_results["candidate_recall"],
        notes="Initial end-to-end baseline on validation sample",
    )
    print_experiment_leaderboard()


def run_test_pipeline(
    data_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "outputs/model.joblib",
    threshold: float = 0.50,
) -> None:
    """Execute competition test inference and generate official submission files."""
    print("=" * 70)
    print("             AMAZON ML ENTITY RESOLUTION -- TEST INFERENCE              ")
    print("=" * 70)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run validation pipeline first to train the model!")

    model = joblib.load(model_path)
    print(f"Loaded trained matching model from: {model_path}")

    # 1. Load Test Data
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s2_path = os.path.join(data_dir, "test_source2.tsv")
    s3_path = os.path.join(data_dir, "test_source3.tsv")

    print(f"Loading test datasets from {data_dir}...")
    df_s1 = load_source_data(s1_path)
    df_s2 = load_source_data(s2_path)
    df_s3 = load_source_data(s3_path)

    # 2. Blocking / Candidate Pairs
    print("\n--- Step 1: Candidate Generation ---")
    df_candidates, pair_list = generate_candidate_pairs_fast(df_s1, df_s2, df_s3)
    cand_file = os.path.join(output_dir, "candidate_pairs.tsv")
    save_results(df_candidates, cand_file, col_name="candidate_entity_ids")

    # 3. Feature Extraction
    print("\n--- Step 2: Feature Extraction ---", flush=True)
    print("Preparing entity lookup tables with normalization caching...", flush=True)
    s1_dict = prepare_entity_lookup(df_s1)
    s23_dict = prepare_entity_lookup(df_s2)
    s23_dict.update(prepare_entity_lookup(df_s3))
    df_pairs, X = extract_features_for_pairs(pair_list, s1_dict, s23_dict)

    # 4. Model Inference
    print("\n--- Step 3: Model Scoring ---")
    scores = model.predict_probabilities(X)
    df_pairs["score"] = scores

    # 5. Generate matching_results.tsv using validated threshold
    print(f"\n--- Step 4: Applying Decision Threshold ({threshold:.2f}) ---")
    match_file = os.path.join(output_dir, "matching_results.tsv")
    all_s1_ids = set(df_s1["entity_id"])
    generate_predictions_at_threshold(df_pairs, all_s1_ids, threshold, match_file)

    # 6. Submission Validation
    print("\n--- Step 5: Submission Pre-Flight Validation ---")
    validate_submission_files(
        matching_file=match_file,
        candidate_file=cand_file,
        test_dir=data_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Run Amazon ML Entity Resolution End-to-End Pipeline.")
    parser.add_argument("--mode", choices=["val", "test"], default="val", help="Run mode: 'val' for validation, 'test' for official test inference")
    parser.add_argument("--data-dir", default=None, help="Custom data directory")
    parser.add_argument("--output-dir", default=None, help="Custom output directory")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification threshold for test inference")
    args = parser.parse_args()

    if args.mode == "val":
        data_dir = args.data_dir or "dataset/val_sample"
        output_dir = args.output_dir or "outputs/val_run"
        run_validation_pipeline(data_dir=data_dir, output_dir=output_dir)
    else:
        data_dir = args.data_dir or "dataset/test"
        output_dir = args.output_dir or "output"
        run_test_pipeline(data_dir=data_dir, output_dir=output_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()
