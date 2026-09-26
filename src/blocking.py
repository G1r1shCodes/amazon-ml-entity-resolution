"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
"""

from typing import Dict, List, Set, Tuple
import pandas as pd
from src.normalize import normalize_address, normalize_business_name


def generate_blocking_keys(name: str, address: str, country: str) -> List[str]:
    """Generate blocking index keys for a record.

    Example blocking keys:
    - Country + First 3 chars of normalized name
    - Country + First token of normalized name
    """
    norm_name = normalize_business_name(name)
    norm_addr = normalize_address(address)
    tokens = norm_name.split()

    keys = []
    if norm_name and country:
        # Prefix key
        keys.append(f"{country}_{norm_name[:3]}")
        # First word key
        if tokens:
            keys.append(f"{country}_{tokens[0]}")
    return keys


def build_blocking_index(df_source: pd.DataFrame) -> Dict[str, List[str]]:
    """Build an inverted index mapping blocking_key -> list of entity_ids."""
    index: Dict[str, List[str]] = {}

    for _, row in df_source.iterrows():
        entity_id = row["entity_id"]
        keys = generate_blocking_keys(
            row["business_name"], row["business_address"], row["country"]
        )
        for key in keys:
            if key not in index:
                index[key] = []
            index[key].append(entity_id)

    return index


def generate_candidate_pairs(
    df_s1: pd.DataFrame, df_s2: pd.DataFrame, df_s3: pd.DataFrame
) -> pd.DataFrame:
    """Generate candidate entity pairs for each Source 1 record.

    Returns DataFrame with columns: ['source1_entity_id', 'candidate_entity_ids']
    """
    print("Building blocking index for Source 2 and Source 3...")
    index_s2 = build_blocking_index(df_s2)
    index_s3 = build_blocking_index(df_s3)

    results = []

    print("Retrieving candidates for Source 1 records...")
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        keys = generate_blocking_keys(
            row["business_name"], row["business_address"], row["country"]
        )

        candidates: Set[str] = set()
        for key in keys:
            if key in index_s2:
                candidates.update(index_s2[key])
            if key in index_s3:
                candidates.update(index_s3[key])

        candidate_str = ",".join(sorted(candidates))
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": candidate_str
        })

    return pd.DataFrame(results)
