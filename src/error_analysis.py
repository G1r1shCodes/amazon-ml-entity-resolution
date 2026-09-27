"""Error Analysis and Diagnostic Module for Entity Resolution.

Role: Person 3 (Evaluation, Validation & Error Diagnostics)

Categorizes errors into 4 key types:
1. False Positives (Incorrectly merged distinct entities)
2. False Negatives (Missed true entity matches)
3. Blocking Failures (True matches never included in candidate_pairs.tsv by Person 1)
4. Model Scoring Failures (True match present in candidate set, but scored below threshold by Person 2)
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz
from src.evaluate import load_mapping_from_tsv


def load_entity_lookup(file_path: str) -> Dict[str, Dict[str, str]]:
    """Load entity metadata (name, address, country) into a fast lookup dict."""
    if not os.path.isfile(file_path):
        return {}
    df = pd.read_csv(file_path, sep="\t", dtype=str)
    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        eid = str(row["entity_id"]).strip()
        lookup[eid] = {
            "business_name": str(row.get("business_name", "")),
            "business_address": str(row.get("business_address", "")),
            "country": str(row.get("country", "")),
        }
    return lookup


def analyze_errors(
    predictions_map: Dict[str, Set[str]],
    ground_truth_map: Dict[str, Set[str]],
    candidates_map: Optional[Dict[str, Set[str]]],
    s1_lookup: Dict[str, Dict[str, str]],
    s23_lookup: Dict[str, Dict[str, str]],
    max_records: int = 50,
) -> Dict[str, List[Dict[str, any]]]:
    """Perform detailed error categorization and similarity diagnostics."""
    false_positives = []
    false_negatives_blocking = []
    false_negatives_model = []

    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())
        cand_set = candidates_map.get(s1_id, set()) if candidates_map else set()

        s1_info = s1_lookup.get(s1_id, {"business_name": "", "business_address": "", "country": ""})

        # 1. False Positives: In Pred but NOT in True
        fp_ids = pred_set - true_set
        for target_id in fp_ids:
            target_info = s23_lookup.get(target_id, {"business_name": "", "business_address": "", "country": ""})
            name_sim = fuzz.token_sort_ratio(s1_info["business_name"], target_info["business_name"])
            addr_sim = fuzz.token_sort_ratio(s1_info["business_address"], target_info["business_address"])

            false_positives.append({
                "source1_id": s1_id,
                "target_id": target_id,
                "error_type": "False Positive (Spurious Match)",
                "s1_name": s1_info["business_name"],
                "target_name": target_info["business_name"],
                "name_similarity": name_sim,
                "s1_address": s1_info["business_address"],
                "target_address": target_info["business_address"],
                "addr_similarity": addr_sim,
                "country": s1_info["country"],
            })

        # 2. False Negatives: In True but NOT in Pred
        fn_ids = true_set - pred_set
        for target_id in fn_ids:
            target_info = s23_lookup.get(target_id, {"business_name": "", "business_address": "", "country": ""})
            name_sim = fuzz.token_sort_ratio(s1_info["business_name"], target_info["business_name"])
            addr_sim = fuzz.token_sort_ratio(s1_info["business_address"], target_info["business_address"])

            if candidates_map and target_id not in cand_set:
                # Type 3: Missing candidate (Blocking failure)
                false_negatives_blocking.append({
                    "source1_id": s1_id,
                    "target_id": target_id,
                    "error_type": "Missing Candidate (Person 1 Blocking Issue)",
                    "s1_name": s1_info["business_name"],
                    "target_name": target_info["business_name"],
                    "name_similarity": name_sim,
                    "s1_address": s1_info["business_address"],
                    "target_address": target_info["business_address"],
                    "addr_similarity": addr_sim,
                    "country": s1_info["country"],
                })
            else:
                # Type 4: Candidate present but model missed it (Model/Threshold issue)
                false_negatives_model.append({
                    "source1_id": s1_id,
                    "target_id": target_id,
                    "error_type": "Model Rejection (Person 2 Model/Threshold Issue)",
                    "s1_name": s1_info["business_name"],
                    "target_name": target_info["business_name"],
                    "name_similarity": name_sim,
                    "s1_address": s1_info["business_address"],
                    "target_address": target_info["business_address"],
                    "addr_similarity": addr_sim,
                    "country": s1_info["country"],
                })

    return {
        "false_positives": false_positives,
        "false_negatives_blocking": false_negatives_blocking,
        "false_negatives_model": false_negatives_model,
    }


def print_error_summary(
    errors: Dict[str, List[Dict[str, any]]],
    n_sample: int = 3,
) -> None:
    """Print an executive summary of diagnosed errors."""
    n_fp = len(errors["false_positives"])
    n_fn_block = len(errors["false_negatives_blocking"])
    n_fn_model = len(errors["false_negatives_model"])
    total_fn = n_fn_block + n_fn_model

    print("\n" + "=" * 70)
    print("                 ERROR ANALYSIS & DIAGNOSTICS REPORT                  ")
    print("=" * 70)
    print(f"Total False Positives (Precision Penalty) : {n_fp:,}")
    print(f"Total False Negatives (Recall Penalty)    : {total_fn:,}")
    if n_fn_block or n_fn_model:
        print(f"  - Missed by Blocking (Person 1)         : {n_fn_block:,} ({n_fn_block/total_fn*100:.1f}%)" if total_fn > 0 else "")
        print(f"  - Rejected by Model  (Person 2)         : {n_fn_model:,} ({n_fn_model/total_fn*100:.1f}%)" if total_fn > 0 else "")
    print("-" * 70)

    # Show Sample False Positives
    if errors["false_positives"]:
        print(f"\n[SAMPLE FALSE POSITIVES (Why Did the System Mistakenly Merge?)]")
        for i, item in enumerate(errors["false_positives"][:n_sample], 1):
            print(f" {i}. S1: {item['source1_id']} vs Target: {item['target_id']} (Country: {item['country']})")
            print(f"    S1 Name     : {item['s1_name']}")
            print(f"    Target Name : {item['target_name']} (Name Sim: {item['name_similarity']}%)")
            print(f"    S1 Address  : {item['s1_address']}")
            print(f"    Target Addr : {item['target_address']} (Addr Sim: {item['addr_similarity']}%)")

    # Show Sample Missing Candidates (Blocking Issue)
    if errors["false_negatives_blocking"]:
        print(f"\n[SAMPLE BLOCKING FAILURES (True Matches Not In candidate_pairs.tsv)]")
        for i, item in enumerate(errors["false_negatives_blocking"][:n_sample], 1):
            print(f" {i}. S1: {item['source1_id']} vs Target: {item['target_id']} (Country: {item['country']})")
            print(f"    S1 Name     : {item['s1_name']}")
            print(f"    Target Name : {item['target_name']} (Name Sim: {item['name_similarity']}%)")
            print(f"    S1 Address  : {item['s1_address']}")
            print(f"    Target Addr : {item['target_address']} (Addr Sim: {item['addr_similarity']}%)")

    # Show Sample Model Rejections
    if errors["false_negatives_model"]:
        print(f"\n[SAMPLE MODEL FAILURES (Present in Candidates but Scored Low)]")
        for i, item in enumerate(errors["false_negatives_model"][:n_sample], 1):
            print(f" {i}. S1: {item['source1_id']} vs Target: {item['target_id']} (Country: {item['country']})")
            print(f"    S1 Name     : {item['s1_name']}")
            print(f"    Target Name : {item['target_name']} (Name Sim: {item['name_similarity']}%)")
            print(f"    S1 Address  : {item['s1_address']}")
            print(f"    Target Addr : {item['target_address']} (Addr Sim: {item['addr_similarity']}%)")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze and diagnose entity resolution prediction errors.")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions TSV")
    parser.add_argument("--ground-truth", "-g", required=True, help="Path to ground truth TSV")
    parser.add_argument("--source1", "-s1", required=True, help="Path to Source 1 TSV")
    parser.add_argument("--source2", "-s2", required=True, help="Path to Source 2 TSV")
    parser.add_argument("--source3", "-s3", required=True, help="Path to Source 3 TSV")
    parser.add_argument("--candidates", "-c", default=None, help="Optional path to candidate_pairs.tsv")
    parser.add_argument("--output-dir", "-o", default="outputs/error_analysis", help="Directory to save error diagnosis CSVs")
    args = parser.parse_args()

    print(f"Loading predictions from: {args.predictions}")
    preds = load_mapping_from_tsv(args.predictions, "source1_entity_id", "matched_entity_ids")

    print(f"Loading ground truth from: {args.ground_truth}")
    gt = load_mapping_from_tsv(args.ground_truth, "source1_entity_id", "matched_entity_ids")

    candidates = None
    if args.candidates and os.path.isfile(args.candidates):
        print(f"Loading candidate pairs from: {args.candidates}")
        candidates = load_mapping_from_tsv(args.candidates, "source1_entity_id", "candidate_entity_ids")

    print("Loading entity metadata for diagnostic reports...")
    s1_lookup = load_entity_lookup(args.source1)
    s23_lookup = {}
    s23_lookup.update(load_entity_lookup(args.source2))
    s23_lookup.update(load_entity_lookup(args.source3))

    errors = analyze_errors(preds, gt, candidates, s1_lookup, s23_lookup)
    print_error_summary(errors)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        if errors["false_positives"]:
            pd.DataFrame(errors["false_positives"]).to_csv(
                os.path.join(args.output_dir, "false_positives.csv"), index=False
            )
        if errors["false_negatives_blocking"]:
            pd.DataFrame(errors["false_negatives_blocking"]).to_csv(
                os.path.join(args.output_dir, "false_negatives_blocking.csv"), index=False
            )
        if errors["false_negatives_model"]:
            pd.DataFrame(errors["false_negatives_model"]).to_csv(
                os.path.join(args.output_dir, "false_negatives_model.csv"), index=False
            )
        print(f"Error diagnostics exported to: {args.output_dir}")


if __name__ == "__main__":
    main()
