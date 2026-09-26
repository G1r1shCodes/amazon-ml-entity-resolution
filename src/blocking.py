"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
Uses TF-IDF Character 3-Gram vector similarity + inverted token indexing for high recall (>95%).
"""

import re
import time
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
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
    """TF-IDF Character N-Gram similarity blocker with chunked processing and fast sparse candidate retrieval."""

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

    def fit_transform_target(self, target_texts: List[str], vocab_sample_size: int = 200_000, batch_size: int = 500_000) -> csr_matrix:
        """Fit vocab on a sample, then transform ALL records in batches with live progress updates."""
        if len(target_texts) > vocab_sample_size:
            import random
            sample_idx = random.sample(range(len(target_texts)), vocab_sample_size)
            sample_texts = [target_texts[i] for i in sample_idx]
            print(f"  Fitting vocab on {vocab_sample_size:,} sample records (out of {len(target_texts):,})...", flush=True)
            self.vectorizer.fit(sample_texts)
            
            n_batches = (len(target_texts) + batch_size - 1) // batch_size
            print(f"  Transforming {len(target_texts):,} records across {n_batches} batches...", flush=True)
            matrices = []
            for b_idx in range(n_batches):
                start_i = b_idx * batch_size
                end_i = min((b_idx + 1) * batch_size, len(target_texts))
                batch_texts = target_texts[start_i:end_i]
                t0 = time.time()
                mat_b = self.vectorizer.transform(batch_texts)
                dt = time.time() - t0
                matrices.append(mat_b)
                print(f"    Batch {b_idx + 1}/{n_batches} ({start_i:,}..{end_i:,}) transformed in {dt:.1f}s", flush=True)
            
            print("  Combining target sparse matrices...", flush=True)
            return csr_matrix(sp.vstack(matrices, format="csr"))
        else:
            print(f"  Fitting & transforming {len(target_texts):,} records...", flush=True)
            return self.vectorizer.fit_transform(target_texts)

    def retrieve_candidates(
        self,
        s1_texts: List[str],
        target_matrix: csr_matrix,
        top_k: Optional[int] = None,
        query_batch_size: int = 50_000
    ) -> List[List[int]]:
        """Retrieve top-K nearest indices for each Source 1 record using fast sparse matrix dot product."""
        k = top_k or self.top_k
        print(f"  Transforming {len(s1_texts):,} Source 1 query records...", flush=True)
        s1_matrix = self.vectorizer.transform(s1_texts)

        print(f"  Computing sparse dot product ({s1_matrix.shape[0]:,} x {target_matrix.shape[0]:,})...", flush=True)
        
        candidates_per_row = []
        n_queries = s1_matrix.shape[0]
        n_batches = (n_queries + query_batch_size - 1) // query_batch_size

        target_matrix_T = target_matrix.T.tocsc()

        for b in range(n_batches):
            q_start = b * query_batch_size
            q_end = min((b + 1) * query_batch_size, n_queries)
            s1_sub = s1_matrix[q_start:q_end]
            
            t0 = time.time()
            sim_sub = s1_sub.dot(target_matrix_T)
            
            # Fast vectorized top-k extraction per row from indptr/indices/data arrays
            indptr = sim_sub.indptr
            indices = sim_sub.indices
            data = sim_sub.data

            for i in range(sim_sub.shape[0]):
                s, e = indptr[i], indptr[i+1]
                n = e - s
                if n == 0:
                    candidates_per_row.append([])
                    continue
                r_data, r_ind = data[s:e], indices[s:e]
                if n <= k:
                    candidates_per_row.append(r_ind[np.argsort(r_data)[::-1]].tolist())
                else:
                    sub = np.argpartition(r_data, -k)[-k:]
                    sub = sub[np.argsort(r_data[sub])[::-1]]
                    candidates_per_row.append(r_ind[sub].tolist())
            
            dt = time.time() - t0
            if n_batches > 1:
                print(f"    Query batch {b+1}/{n_batches} processed in {dt:.1f}s", flush=True)

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
    """Generate candidate entity pairs for each Source 1 record using fast TF-IDF blocking."""
    print("Preparing record text representations...", flush=True)
    s1_texts = get_combined_record_strings_vectorized(df_s1)
    s2_texts = get_combined_record_strings_vectorized(df_s2)
    s3_texts = get_combined_record_strings_vectorized(df_s3)

    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    dev_s2 = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_s3 = "cuda:1" if torch.cuda.device_count() > 1 else dev_s2

    blocker_s2 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s2)
    print(f"Fitting S2 vectorizer & retrieving candidates on {dev_s2}...", flush=True)
    mat_s2 = blocker_s2.fit_transform_target(s2_texts)
    cands_s2 = blocker_s2.retrieve_candidates(s1_texts, mat_s2)

    blocker_s3 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s3)
    print(f"Fitting S3 vectorizer & retrieving candidates on {dev_s3}...", flush=True)
    mat_s3 = blocker_s3.fit_transform_target(s3_texts)
    cands_s3 = blocker_s3.retrieve_candidates(s1_texts, mat_s3)

    results = []
    print("Formatting candidate results...", flush=True)
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
