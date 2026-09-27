"""Feature Engineering Module for Entity Resolution.

Assigned to: Person 2 (similarity features & training pairs)
Computes pairwise similarity features between Source 1 records and Candidate records.

Batch processing: `extract_batch_features()` computes features for all candidate pairs
at once using vectorized string operations and rapidfuzz batch scoring.
"""

import time
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from src.normalize import clean_text, normalize_address, normalize_business_name, pd_isna


# ---------------------------------------------------------------------------
# Feature names (order matters — must match columns in feature matrix)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "name_jaccard",
    "name_levenshtein",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "addr_jaccard",
    "addr_levenshtein",
    "addr_partial_ratio",
    "same_country",
    "name_len_ratio",
    "addr_len_ratio",
]


# ---------------------------------------------------------------------------
# Single-pair feature computation
# ---------------------------------------------------------------------------

def jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token-based Jaccard similarity between two strings."""
    set1 = set(clean_text(str1).split())
    set2 = set(clean_text(str2).split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union)


def _safe_len_ratio(a: str, b: str) -> float:
    """Length ratio: min(len(a), len(b)) / max(len(a), len(b))."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def compute_pair_features(
    name1: any,
    name2: any,
    addr1: str = "",
    addr2: str = "",
    country1: str = "",
    country2: str = "",
) -> np.ndarray:
    """Compute feature vector for a single S1-candidate pair. Supports dict inputs as well."""
    if isinstance(name1, dict):
        rec1 = name1
        rec2 = name2
        n1_raw = str(rec1.get("business_name", ""))
        n2_raw = str(rec2.get("business_name", ""))
        a1_raw = str(rec1.get("business_address", ""))
        a2_raw = str(rec2.get("business_address", ""))
        country1 = str(rec1.get("country", ""))
        country2 = str(rec2.get("country", ""))
        n1_clean = rec1.get("norm_name") or normalize_business_name(n1_raw)[0]
        n2_clean = rec2.get("norm_name") or normalize_business_name(n2_raw)[0]
        a1_clean = rec1.get("norm_addr") or normalize_address(a1_raw)
        a2_clean = rec2.get("norm_addr") or normalize_address(a2_raw)
        n1_set = rec1.get("name_token_set_pre")
        n2_set = rec2.get("name_token_set_pre")
        if n1_set is not None and n2_set is not None:
            u1 = len(n1_set | n2_set)
            name_jac = float(len(n1_set & n2_set) / u1) if u1 > 0 else 0.0
        else:
            name_jac = jaccard_similarity(n1_raw, n2_raw)

        a1_set = rec1.get("addr_token_set_pre")
        a2_set = rec2.get("addr_token_set_pre")
        if a1_set is not None and a2_set is not None:
            u2 = len(a1_set | a2_set)
            addr_jac = float(len(a1_set & a2_set) / u2) if u2 > 0 else 0.0
        else:
            addr_jac = jaccard_similarity(a1_raw, a2_raw)
    else:
        name1 = str(name1)
        name2 = str(name2)
        n1_clean, _ = normalize_business_name(name1)
        n2_clean, _ = normalize_business_name(name2)
        a1_clean = normalize_address(addr1)
        a2_clean = normalize_address(addr2)
        name_jac = jaccard_similarity(name1, name2)
        addr_jac = jaccard_similarity(addr1, addr2)

    return np.array([
        name_jac,
        fuzz.ratio(n1_clean, n2_clean) / 100.0,
        fuzz.partial_ratio(n1_clean, n2_clean) / 100.0,
        fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0,
        fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0,
        addr_jac,
        fuzz.ratio(a1_clean, a2_clean) / 100.0,
        fuzz.partial_ratio(a1_clean, a2_clean) / 100.0,
        1.0 if str(country1).lower() == str(country2).lower() and str(country1).strip() != "" else 0.0,
        _safe_len_ratio(n1_clean, n2_clean),
        _safe_len_ratio(a1_clean, a2_clean),
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch feature extraction
# ---------------------------------------------------------------------------

def _build_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build entity_id -> {business_name, business_address, country} lookup dict."""
    records = {}
    for _, row in df.iterrows():
        records[str(row["entity_id"])] = {
            "business_name": str(row.get("business_name", "")),
            "business_address": str(row.get("business_address", "")),
            "country": str(row.get("country", "")),
        }
    return records


def extract_batch_features(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    candidates_df: pd.DataFrame,
    gt_map: Optional[Dict[str, Set[str]]] = None,
    max_pairs_per_s1: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Extract features for all S1-candidate pairs in batch.

    Args:
        df_s1: Source 1 records.
        df_s2: Source 2 records (for looking up candidate details).
        df_s3: Source 3 records (for looking up candidate details).
        candidates_df: DataFrame with columns [source1_entity_id, candidate_entity_ids].
        gt_map: Ground truth mapping s1_id -> set of matched entity IDs.
                 If provided, labels are generated (1 = match, 0 = non-match).
                 If None, labels are all -1 (inference mode).
        max_pairs_per_s1: Limit candidates per S1 record (for memory control).

    Returns:
        X: Feature matrix of shape (n_pairs, n_features).
        y: Label vector of shape (n_pairs,). 1=match, 0=non-match, -1=unknown.
        pair_ids: List of (s1_id, candidate_id) tuples for each row.
    """
    t0 = time.time()
    print("Building entity lookup tables...", flush=True)

    # Build lookups — only for entities that appear in candidates or S1 sample
    s1_ids_needed = set(candidates_df["source1_entity_id"].tolist())
    s1_subset = df_s1[df_s1["entity_id"].isin(s1_ids_needed)]
    s1_lookup = _build_lookup(s1_subset)

    # Collect all candidate IDs to filter S2/S3 lookups
    all_cand_ids = set()
    for _, row in candidates_df.iterrows():
        cand_str = str(row.get("candidate_entity_ids", ""))
        for c in cand_str.split(","):
            c = c.strip()
            if c:
                all_cand_ids.add(c)

    s2_subset = df_s2[df_s2["entity_id"].isin(all_cand_ids)]
    s3_subset = df_s3[df_s3["entity_id"].isin(all_cand_ids)]
    cand_lookup = {**_build_lookup(s2_subset), **_build_lookup(s3_subset)}

    print(f"  S1: {len(s1_lookup):,}, Candidates: {len(cand_lookup):,} "
          f"(built in {time.time()-t0:.1f}s)", flush=True)

    # Expand candidate pairs into flat list
    print("Expanding candidate pairs...", flush=True)
    pair_ids: List[Tuple[str, str]] = []
    for _, row in candidates_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        cand_str = str(row.get("candidate_entity_ids", ""))
        if not cand_str.strip():
            continue
        cand_ids = [c.strip() for c in cand_str.split(",") if c.strip()]
        if max_pairs_per_s1 and len(cand_ids) > max_pairs_per_s1:
            cand_ids = cand_ids[:max_pairs_per_s1]
        for cid in cand_ids:
            pair_ids.append((s1_id, cid))

    n_pairs = len(pair_ids)
    n_features = len(FEATURE_NAMES)
    print(f"  {n_pairs:,} total pairs to featurize", flush=True)

    # Allocate feature matrix and label vector
    X = np.zeros((n_pairs, n_features), dtype=np.float32)
    y = np.full(n_pairs, -1, dtype=np.int8)

    # Compute features for each pair
    print("Computing features...", flush=True)
    t1 = time.time()
    progress_interval = max(1, n_pairs // 10)

    for i, (s1_id, cand_id) in enumerate(pair_ids):
        s1_rec = s1_lookup.get(s1_id)
        cand_rec = cand_lookup.get(cand_id)

        if s1_rec is None or cand_rec is None:
            continue

        X[i] = compute_pair_features(
            s1_rec["business_name"], cand_rec["business_name"],
            s1_rec["business_address"], cand_rec["business_address"],
            s1_rec["country"], cand_rec["country"],
        )

        # Generate label from ground truth
        if gt_map is not None:
            matched = gt_map.get(s1_id, set())
            y[i] = 1 if cand_id in matched else 0

        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - t1
            rate = (i + 1) / elapsed
            remaining = (n_pairs - i - 1) / rate
            print(f"  {i+1:,}/{n_pairs:,} pairs ({rate:.0f} pairs/s, "
                  f"~{remaining:.0f}s remaining)", flush=True)

    elapsed = time.time() - t0
    print(f"✅ Feature extraction complete: {n_pairs:,} pairs x {n_features} features "
          f"in {elapsed:.1f}s", flush=True)

    if gt_map is not None:
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"  Labels: {n_pos:,} positive, {n_neg:,} negative "
              f"(ratio 1:{n_neg//max(n_pos,1)})", flush=True)

    return X, y, pair_ids
