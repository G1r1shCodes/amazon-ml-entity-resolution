"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
Uses TF-IDF Character 3-Gram vector similarity + GPU-accelerated (CuPy cuSPARSE) sparse matmul.
Falls back to chunked SciPy sparse CPU matmul if CuPy is unavailable.
"""

import re
import time
from typing import Dict, List, Tuple, Optional
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


def _try_import_cupy():
    """Try to import CuPy + CuPy sparse. Returns (cp, csp) or (None, None)."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        # Quick sanity check
        cp.array([1.0])
        return cp, csp
    except Exception:
        return None, None


def _merge_topk(running_scores: np.ndarray, running_idx: np.ndarray,
                new_scores: np.ndarray, new_idx: np.ndarray, k: int):
    """Merge running top-k with new chunk top-k. All arrays shape (n_queries, k)."""
    combined_scores = np.concatenate([running_scores, new_scores], axis=1)
    combined_idx = np.concatenate([running_idx, new_idx], axis=1)
    n_queries = combined_scores.shape[0]
    take = min(k, combined_scores.shape[1])
    best_pos = np.argpartition(combined_scores, -take, axis=1)[:, -take:]
    return (combined_scores[np.arange(n_queries)[:, None], best_pos],
            combined_idx[np.arange(n_queries)[:, None], best_pos])


class TFIDFBlocker:
    """TF-IDF Character N-Gram blocker with CuPy GPU sparse matmul acceleration."""

    def __init__(self, top_k: int = 30, ngram_range: Tuple[int, int] = (3, 3),
                 device: str = "cuda:0", max_features: int = 100000):
        self.top_k = top_k
        self.device = device
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            min_df=5,
            max_features=max_features,
            dtype=np.float32
        )

    def fit_transform_target(self, target_texts: List[str],
                             vocab_sample_size: int = 200_000,
                             batch_size: int = 500_000) -> csr_matrix:
        """Fit vocab on a sample, then transform all records in batches."""
        if len(target_texts) > vocab_sample_size:
            import random
            sample_idx = random.sample(range(len(target_texts)), vocab_sample_size)
            print(f"  Fitting vocab on {vocab_sample_size:,} sample records "
                  f"(out of {len(target_texts):,})...", flush=True)
            self.vectorizer.fit([target_texts[i] for i in sample_idx])

            n_batches = (len(target_texts) + batch_size - 1) // batch_size
            print(f"  Transforming {len(target_texts):,} records in {n_batches} batches...", flush=True)
            mats = []
            for b in range(n_batches):
                s, e = b * batch_size, min((b + 1) * batch_size, len(target_texts))
                t0 = time.time()
                mats.append(self.vectorizer.transform(target_texts[s:e]))
                print(f"    Batch {b+1}/{n_batches} ({s:,}..{e:,}) done in {time.time()-t0:.1f}s", flush=True)

            print("  Stacking batches...", flush=True)
            return csr_matrix(sp.vstack(mats, format="csr"))
        else:
            print(f"  Fitting & transforming {len(target_texts):,} records...", flush=True)
            return self.vectorizer.fit_transform(target_texts)

    def retrieve_candidates(self, s1_texts: List[str], target_matrix: csr_matrix,
                            top_k: Optional[int] = None,
                            target_chunk_size: int = 100_000) -> List[List[int]]:
        """Retrieve top-K candidates using CuPy GPU sparse matmul (falls back to SciPy CPU).

        Chunks the 5M target matrix into slices so result matrices stay small:
          GPU: (n_queries × chunk_size) as dense float32 in VRAM
          CPU: same but in RAM.
        """
        k = top_k or self.top_k
        n_queries = len(s1_texts)
        n_targets = target_matrix.shape[0]
        n_chunks = (n_targets + target_chunk_size - 1) // target_chunk_size

        print(f"  Transforming {n_queries:,} S1 query records...", flush=True)
        s1_matrix = self.vectorizer.transform(s1_texts)  # (n_queries × vocab) sparse

        # Try GPU via CuPy
        cp, csp = _try_import_cupy()
        use_gpu = (cp is not None) and torch.cuda.is_available()

        device_label = self.device if use_gpu else "CPU (SciPy)"
        print(f"  Running chunked top-{k} retrieval on {device_label}: "
              f"{n_queries:,} queries × {n_targets:,} targets in {n_chunks} chunks of {target_chunk_size:,}...",
              flush=True)

        # Pre-upload s1 to GPU once (stays there across all chunks)
        if use_gpu:
            try:
                s1_gpu = csp.csr_matrix(s1_matrix.astype(np.float32))
                print(f"  ✅ S1 query matrix uploaded to GPU VRAM "
                      f"({s1_gpu.nnz:,} nnz, {s1_gpu.data.nbytes / 1e6:.1f} MB)", flush=True)
            except Exception as ex:
                print(f"  ⚠️ GPU upload failed ({ex}), falling back to CPU.", flush=True)
                use_gpu = False

        # Running top-k buffers
        top_scores = np.full((n_queries, k), -1.0, dtype=np.float32)
        top_indices = np.full((n_queries, k), -1, dtype=np.int32)

        total_t0 = time.time()
        for c_idx in range(n_chunks):
            c_start = c_idx * target_chunk_size
            c_end = min(c_start + target_chunk_size, n_targets)
            chunk_scipy = target_matrix[c_start:c_end]  # (chunk_size × vocab)

            t0 = time.time()
            if use_gpu:
                try:
                    chunk_gpu = csp.csr_matrix(chunk_scipy.astype(np.float32))
                    # GPU sparse matmul: (n_queries × vocab) @ (vocab × chunk_size) → (n_queries × chunk_size)
                    sim_gpu = (s1_gpu @ chunk_gpu.T).toarray()   # cuSPARSE → dense on GPU
                    sim = cp.asnumpy(sim_gpu)                     # pull to CPU numpy
                    del sim_gpu, chunk_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    label = "GPU"
                except Exception as ex:
                    print(f"  ⚠️ GPU chunk failed ({ex}), switching to CPU for this chunk.", flush=True)
                    sim = s1_matrix.dot(chunk_scipy.T).toarray()
                    label = "CPU"
            else:
                sim = s1_matrix.dot(chunk_scipy.T).toarray()
                label = "CPU"

            dt = time.time() - t0

            # Find top-k within this chunk for every query
            local_k = min(k, sim.shape[1])
            local_best_pos = np.argpartition(sim, -local_k, axis=1)[:, -local_k:]
            local_scores = sim[np.arange(n_queries)[:, None], local_best_pos]
            local_idx = (local_best_pos + c_start).astype(np.int32)

            # Merge with running top-k
            top_scores, top_indices = _merge_topk(top_scores, top_indices,
                                                   local_scores, local_idx, k)

            elapsed = time.time() - total_t0
            print(f"    [{label}] Chunk {c_idx+1}/{n_chunks} "
                  f"({c_start:,}..{c_end:,}) in {dt:.1f}s  [total {elapsed:.0f}s]", flush=True)

        print(f"  ✅ All chunks done in {time.time()-total_t0:.1f}s total.", flush=True)

        # Build output: filter out unfilled -1 slots
        return [top_indices[i][top_indices[i] >= 0].tolist() for i in range(n_queries)]


def get_combined_record_strings_vectorized(df: pd.DataFrame) -> List[str]:
    """Fast vectorized text preparation for large DataFrames."""
    names = df["business_name"].fillna("").astype(str).str.lower()
    addrs = df["business_address"].fillna("").astype(str).str.lower()
    countries = df["country"].fillna("").astype(str).str.lower()
    return (names + " " + addrs + " " + countries).tolist()


def generate_candidate_pairs(df_s1: pd.DataFrame, df_s2: pd.DataFrame,
                             df_s3: pd.DataFrame, top_k_per_source: int = 25) -> pd.DataFrame:
    """Generate candidate entity pairs for each Source 1 record."""
    print("Preparing record text representations...", flush=True)
    s1_texts = get_combined_record_strings_vectorized(df_s1)
    s2_texts = get_combined_record_strings_vectorized(df_s2)
    s3_texts = get_combined_record_strings_vectorized(df_s3)

    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    dev_s2 = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_s3 = "cuda:1" if torch.cuda.device_count() > 1 else dev_s2

    blocker_s2 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s2)
    print(f"\nFitting S2 vectorizer & retrieving candidates on {dev_s2}...", flush=True)
    mat_s2 = blocker_s2.fit_transform_target(s2_texts)
    cands_s2 = blocker_s2.retrieve_candidates(s1_texts, mat_s2)

    blocker_s3 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s3)
    print(f"\nFitting S3 vectorizer & retrieving candidates on {dev_s3}...", flush=True)
    mat_s3 = blocker_s3.fit_transform_target(s3_texts)
    cands_s3 = blocker_s3.retrieve_candidates(s1_texts, mat_s3)

    print("\nFormatting candidate results...", flush=True)
    results = []
    for idx, s1_id in enumerate(df_s1["entity_id"].tolist()):
        matched = set()
        for s2_idx in cands_s2[idx]:
            matched.add(s2_ids[s2_idx])
        for s3_idx in cands_s3[idx]:
            matched.add(s3_ids[s3_idx])
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(matched))
        })

    return pd.DataFrame(results)
