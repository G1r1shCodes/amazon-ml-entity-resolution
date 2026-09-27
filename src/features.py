"""Feature Engineering Module for Entity Resolution.

Assigned to: Person 2 (similarity features & training pairs)
Computes pairwise similarity features between Source 1 records and Candidate records.
"""

from typing import Dict, Any, List
import pandas as pd
from rapidfuzz import distance, fuzz
from src.normalize import clean_text, normalize_address, normalize_business_name


def jaccard_similarity(str1: str, str2: str) -> float:
    """Compute token-based Jaccard similarity between two strings."""
    set1 = set(clean_text(str1).split())
    set2 = set(clean_text(str2).split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union)


def compute_pair_features(record_s1: pd.Series, record_cand: pd.Series) -> Dict[str, float]:
    """Compute pairwise similarity features between an S1 record and a candidate record."""
    name1 = str(record_s1["business_name"])
    name2 = str(record_cand["business_name"])

    addr1 = str(record_s1["business_address"])
    addr2 = str(record_cand["business_address"])

    country1 = str(record_s1["country"])
    country2 = str(record_cand["country"])

    norm_name1 = record_s1.get("norm_name") or normalize_business_name(name1)
    norm_name2 = record_cand.get("norm_name") or normalize_business_name(name2)

    norm_addr1 = record_s1.get("norm_addr") or normalize_address(addr1)
    norm_addr2 = record_cand.get("norm_addr") or normalize_address(addr2)

    features: Dict[str, float] = {
        # Name similarity features
        "name_jaccard": jaccard_similarity(norm_name1, norm_name2),
        "name_levenshtein": float(fuzz.ratio(norm_name1, norm_name2) / 100.0),
        "name_partial_ratio": float(fuzz.partial_ratio(norm_name1, norm_name2) / 100.0),
        "name_token_sort": float(fuzz.token_sort_ratio(norm_name1, norm_name2) / 100.0),

        # Address similarity features
        "address_jaccard": jaccard_similarity(norm_addr1, norm_addr2),
        "address_levenshtein": float(fuzz.ratio(norm_addr1, norm_addr2) / 100.0),

        # Metadata features
        "same_country": 1.0 if country1.lower() == country2.lower() and country1 != "" else 0.0,
    }

    return features
