"""Validation Split Generator for Fast Local Experimentation.

Assigned to: Person 3 (Evaluation & Validation)

Creates a representative mini-validation dataset (e.g. 10,000 Source 1 entities)
stratified by country, including all true matches from Source 2 & Source 3, plus
a realistic pool of negative distractor records.
"""

import argparse
import os
import random
from typing import Set
import pandas as pd


def create_validation_split(
    data_dir: str = "dataset/train",
    output_dir: str = "dataset/val_sample",
    n_samples: int = 10000,
    distractor_ratio: int = 5,
    random_seed: int = 42,
) -> None:
    """Generate a self-contained validation benchmark.

    Args:
        data_dir: Directory containing train_*.tsv files.
        output_dir: Output directory for validation split files.
        n_samples: Number of Source 1 entities to sample (default: 10,000).
        distractor_ratio: Number of random distractor S2/S3 entities per true match.
        random_seed: Random seed for reproducibility.
    """
    random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    s1_path = os.path.join(data_dir, "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")

    print(f"Loading Source 1 and Ground Truth from {data_dir}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype=str)

    # Merge to align ground truth with Source 1 metadata
    df_merged = df_s1.merge(df_gt, left_on="entity_id", right_on="source1_entity_id", how="inner")

    # Stratified sample by country to maintain US and India proportions
    print(f"Sampling {n_samples:,} Source 1 records (stratified by country)...")
    sample_parts = []
    for country, group in df_merged.groupby("country"):
        n_c = int(n_samples * len(group) / len(df_merged))
        sample_parts.append(group.sample(n=n_c, random_state=random_seed))
    df_sample = pd.concat(sample_parts, ignore_index=True)

    sampled_s1_ids: Set[str] = set(df_sample["entity_id"])
    print(f"Selected {len(sampled_s1_ids):,} Source 1 records.")

    # Identify all required true matches from S2 and S3
    target_s2_ids: Set[str] = set()
    target_s3_ids: Set[str] = set()

    for _, row in df_sample.iterrows():
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        if matched_str.strip():
            for mid in matched_str.split(","):
                mid = mid.strip()
                if mid.startswith("S2-"):
                    target_s2_ids.add(mid)
                elif mid.startswith("S3-"):
                    target_s3_ids.add(mid)

    print(f"True matching entities needed: {len(target_s2_ids):,} from S2, {len(target_s3_ids):,} from S3.")

    # Save validation Source 1
    val_s1 = df_sample[["entity_id", "business_name", "business_address", "country"]]
    val_s1_file = os.path.join(output_dir, "val_source1.tsv")
    val_s1.to_csv(val_s1_file, sep="\t", index=False)
    print(f"Saved: {val_s1_file} ({len(val_s1):,} rows)")

    # Save validation Ground Truth
    val_gt = df_sample[["source1_entity_id", "matched_entity_ids"]]
    val_gt_file = os.path.join(output_dir, "val_ground_truth.tsv")
    val_gt.to_csv(val_gt_file, sep="\t", index=False)
    print(f"Saved: {val_gt_file} ({len(val_gt):,} rows)")

    # Filter Source 2: include all true matches + distractors
    print("Filtering Source 2 records (streaming to conserve RAM)...")
    s2_rows = []
    n_distractors_s2 = len(target_s2_ids) * distractor_ratio
    distractor_pool_s2 = []

    for chunk in pd.read_csv(s2_path, sep="\t", dtype=str, chunksize=100000):
        # Keep true matches
        matches = chunk[chunk["entity_id"].isin(target_s2_ids)]
        if not matches.empty:
            s2_rows.append(matches)
        # Collect candidate distractors (not in target)
        non_matches = chunk[~chunk["entity_id"].isin(target_s2_ids)]
        if len(distractor_pool_s2) < n_distractors_s2 and not non_matches.empty:
            distractor_pool_s2.append(non_matches.sample(n=min(len(non_matches), 5000), random_state=random_seed))

    df_val_s2 = pd.concat(s2_rows + distractor_pool_s2, ignore_index=True).drop_duplicates(subset=["entity_id"])
    val_s2_file = os.path.join(output_dir, "val_source2.tsv")
    df_val_s2.to_csv(val_s2_file, sep="\t", index=False)
    print(f"Saved: {val_s2_file} ({len(df_val_s2):,} rows)")

    # Filter Source 3: include all true matches + distractors
    print("Filtering Source 3 records (streaming to conserve RAM)...")
    s3_rows = []
    n_distractors_s3 = len(target_s3_ids) * distractor_ratio
    distractor_pool_s3 = []

    for chunk in pd.read_csv(s3_path, sep="\t", dtype=str, chunksize=100000):
        # Keep true matches
        matches = chunk[chunk["entity_id"].isin(target_s3_ids)]
        if not matches.empty:
            s3_rows.append(matches)
        # Collect candidate distractors (not in target)
        non_matches = chunk[~chunk["entity_id"].isin(target_s3_ids)]
        if len(distractor_pool_s3) < n_distractors_s3 and not non_matches.empty:
            distractor_pool_s3.append(non_matches.sample(n=min(len(non_matches), 5000), random_state=random_seed))

    df_val_s3 = pd.concat(s3_rows + distractor_pool_s3, ignore_index=True).drop_duplicates(subset=["entity_id"])
    val_s3_file = os.path.join(output_dir, "val_source3.tsv")
    df_val_s3.to_csv(val_s3_file, sep="\t", index=False)
    print(f"Saved: {val_s3_file} ({len(df_val_s3):,} rows)")

    print("\n[SUCCESS] Validation dataset created successfully in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create validation split for local experimentation.")
    parser.add_argument("--data-dir", default="dataset/train", help="Path to training dataset folder")
    parser.add_argument("--output-dir", default="dataset/val_sample", help="Path to save validation split")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of Source 1 entities to sample")
    parser.add_argument("--distractor-ratio", type=int, default=5, help="Distractors ratio per true match")
    args = parser.parse_args()

    create_validation_split(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        distractor_ratio=args.distractor_ratio,
    )
