"""Blocking Audit Script.

Evaluates Person 1's blocking recall ceiling on validation data.
"""

import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.blocking import generate_blocking_keys
from src.evaluate import load_mapping_from_tsv, compute_candidate_recall

def audit_blocking(n_sample: int = 2000):
    print(f"Loading {n_sample:,} Source 1 validation records...")
    df_s1 = pd.read_csv("dataset/val_sample/val_source1.tsv", sep="\t", dtype=str).head(n_sample)
    df_s2 = pd.read_csv("dataset/val_sample/val_source2.tsv", sep="\t", dtype=str)
    df_s3 = pd.read_csv("dataset/val_sample/val_source3.tsv", sep="\t", dtype=str)
    gt_map = load_mapping_from_tsv("dataset/val_sample/val_ground_truth.tsv", "source1_entity_id", "matched_entity_ids")

    t0 = time.time()
    print("Building inverted index for S2 and S3...")
    index = {}
    for eid, name, addr, country in zip(df_s2["entity_id"], df_s2["business_name"].fillna(""), df_s2["business_address"].fillna(""), df_s2["country"].fillna("")):
        for k in generate_blocking_keys(name, addr, country):
            if k not in index:
                index[k] = []
            index[k].append(eid)

    for eid, name, addr, country in zip(df_s3["entity_id"], df_s3["business_name"].fillna(""), df_s3["business_address"].fillna(""), df_s3["country"].fillna("")):
        for k in generate_blocking_keys(name, addr, country):
            if k not in index:
                index[k] = []
            index[k].append(eid)
    print(f"Index built in {time.time() - t0:.2f}s with {len(index):,} distinct blocking keys.")

    t1 = time.time()
    print("Retrieving candidates for S1 sample...")
    cands_map = {}
    total_cands = 0
    for eid, name, addr, country in zip(df_s1["entity_id"], df_s1["business_name"].fillna(""), df_s1["business_address"].fillna(""), df_s1["country"].fillna("")):
        matched_cands = set()
        for k in generate_blocking_keys(name, addr, country):
            if k in index:
                matched_cands.update(index[k])
        cands_map[eid] = matched_cands
        total_cands += len(matched_cands)
    print(f"Candidates generated in {time.time() - t1:.2f}s.")

    sub_gt = {eid: gt_map[eid] for eid in df_s1["entity_id"]}
    rec = compute_candidate_recall(cands_map, sub_gt)

    print("\n" + "=" * 60)
    print("         PERSON 1 BLOCKING QUALITY AUDIT RESULTS           ")
    print("=" * 60)
    print(f"Evaluated Source 1 Entities : {len(df_s1):,}")
    print(f"Total True Matches          : {rec['total_true_matches']:,}")
    print(f"True Matches Captured       : {rec['captured_true_matches']:,}")
    print(f"Candidate Recall Ceiling    : {rec['candidate_recall']*100:.2f}%")
    print(f"Full Coverage Rate          : {rec['full_coverage_rate']*100:.2f}%")
    print(f"Average Candidates per S1   : {total_cands / len(df_s1):.1f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    audit_blocking()
