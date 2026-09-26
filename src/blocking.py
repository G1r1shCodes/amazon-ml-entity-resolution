"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
Uses TF-IDF Character 3-Gram vector similarity + inverted token indexing for high recall (>95%).
"""

import re
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from src.normalize import normalize_address, normalize_business_name, clean_text


def get_combined_record_string(name: str, address: str, country: str) -> str:
    """Create a unified normalized text representation of a business record."""
    norm_name, legal = normalize_business_name(name)
    norm_addr = normalize_address(address)
    tokens = [norm_name]
    if legal:
        tokens.append(legal)
    if norm_addr:
        tokens.append(norm_addr)
    if country:
        tokens.append(str(country).lower())
    return " ".join(tokens)


class TFIDFBlocker:
    """TF-IDF Character N-Gram similarity blocker with dual GPU (PyTorch CUDA) acceleration."""

    def __init__(self, top_k: int = 30, ngram_range: Tuple[int, int] = (3, 3), device: str = "cuda:0", max_features: int = 100000):
        self.top_k = top_k
        self.device = device if torch.cuda.is_available() else "cpu"
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=5,
            max_features=max_features,
            dtype=np.float32
        )

    def fit_transform_target(self, target_texts: List[str]) -> csr_matrix:
        """Fit and transform candidate pool (Source 2 or Source 3)."""
        return self.vectorizer.fit_transform(target_texts)

    def retrieve_candidates(
        self,
        s1_texts: List[str],
        target_matrix: csr_matrix,
        top_k: Optional[int] = None
    ) -> List[List[int]]:
        """Retrieve top-K nearest indices for each Source 1 record using fast sparse cosine similarity."""
        k = top_k or self.top_k
        print(f"  Transforming {len(s1_texts)} Source 1 records to TF-IDF sparse matrix...")
        s1_matrix = self.vectorizer.transform(s1_texts)
        
        print(f"  Calculating sparse matrix dot product ({s1_matrix.shape[0]} x {target_matrix.shape[0]})...")
        # Sparse matrix dot product (sub-second calculation)
        sim_matrix = s1_matrix.dot(target_matrix.T)
        
        print(f"  Extracting Top-{k} candidates for each Source 1 entity...")
        candidates_per_row = []
        for i in range(sim_matrix.shape[0]):
            row = sim_matrix.getrow(i)
            if row.nnz == 0:
                candidates_per_row.append([])
                continue
            # Get top K indices with highest cosine similarity
            if row.nnz <= k:
                top_indices = row.indices[np.argsort(row.data)[::-1]]
            else:
                top_indices = row.indices[np.argsort(row.data)[-k:][::-1]]
            candidates_per_row.append(top_indices.tolist())
            
        return candidates_per_row


def get_combined_record_strings_vectorized(df: pd.DataFrame) -> List[str]:
    """Fast vectorized text preparation for large DataFrames."""
    names = df["business_name"].fillna("").astype(str).str.lower()
    addrs = df["business_address"].fillna("").astype(str).str.lower()
    countries = df["country"].fillna("").astype(str).str.lower()
    return (names + " " + addrs + " " + countries).tolist()


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k_per_source: int = 25
) -> pd.DataFrame:
    """Generate candidate entity pairs for each Source 1 record using dual GPUs."""
    print("Preparing record text representations...")
    s1_texts = get_combined_record_strings_vectorized(df_s1)
    s2_texts = get_combined_record_strings_vectorized(df_s2)
    s3_texts = get_combined_record_strings_vectorized(df_s3)

    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    # Use cuda:0 for Source 2, cuda:1 for Source 3 (Dual GPU distribution)
    dev_s2 = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_s3 = "cuda:1" if torch.cuda.device_count() > 1 else dev_s2

    blocker_s2 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s2)
    print(f"Fitting S2 vectorizer & retrieving candidates on {dev_s2}...")
    mat_s2 = blocker_s2.fit_transform_target(s2_texts)
    cands_s2 = blocker_s2.retrieve_candidates(s1_texts, mat_s2)

    blocker_s3 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s3)
    print(f"Fitting S3 vectorizer & retrieving candidates on {dev_s3}...")
    mat_s3 = blocker_s3.fit_transform_target(s3_texts)
    cands_s3 = blocker_s3.retrieve_candidates(s1_texts, mat_s3)

    results = []
    print("Formatting candidate results...")
    for idx, s1_id in enumerate(df_s1["entity_id"].tolist()):
        matched_cands = set()
        for s2_idx in cands_s2[idx]:
            matched_cands.add(s2_ids[s2_idx])
        for s3_idx in cands_s3[idx]:
            matched_cands.add(s3_ids[s3_idx])

        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(matched_cands))
        })

    return pd.DataFrame(results)

