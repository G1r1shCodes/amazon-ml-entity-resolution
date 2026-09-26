"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.
Uses word n-gram TF-IDF with vocabulary fitted on S1+target combined for high recall.
GPU acceleration via CuPy cuSPARSE sparse matmul.
"""

import time
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

from src.normalize import normalize_address, normalize_business_name, clean_text


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------

def _normalize_name_fast(name: str) -> str:
    """Quick lowercase + strip legal suffixes for vectorized use."""
    import unicodedata, re
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    # Remove common legal suffixes to improve token overlap
    name = re.sub(
        r'\b(llc|ltd|inc|corp|co|plc|gmbh|llp|lp|sa|srl|sl|bv|nv|ag|oy|ab|as|pte|pvt|sas|kk|kg)\b\.?',
        '', name
    )
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def get_combined_record_strings_vectorized(df: pd.DataFrame) -> List[str]:
    """Normalize business names for TF-IDF blocking.
    
    Uses name-only (not address/country) — address is often empty/noisy and
    dilutes the name similarity signal.  Legal suffixes are stripped so that
    'Acme LLC' and 'Acme Inc' produce overlapping tokens.
    """
    names = df["business_name"].fillna("").astype(str)
    return names.apply(_normalize_name_fast).tolist()


# ---------------------------------------------------------------------------
# CuPy helper
# ---------------------------------------------------------------------------

def _try_import_cupy():
    """Try to import CuPy + CuPy sparse. Returns (cp, csp) or (None, None)."""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        cp.array([1.0])          # quick GPU sanity check
        return cp, csp
    except Exception:
        return None, None


def _merge_topk(running_scores: np.ndarray, running_idx: np.ndarray,
                new_scores: np.ndarray, new_idx: np.ndarray, k: int):
    """Merge running top-k with new chunk top-k. All arrays shape (n_queries, k)."""
    combined_s = np.concatenate([running_scores, new_scores], axis=1)
    combined_i = np.concatenate([running_idx,   new_idx],    axis=1)
    n = combined_s.shape[0]
    take = min(k, combined_s.shape[1])
    pos = np.argpartition(combined_s, -take, axis=1)[:, -take:]
    return combined_s[np.arange(n)[:, None], pos], combined_i[np.arange(n)[:, None], pos]


# ---------------------------------------------------------------------------
# TFIDFBlocker
# ---------------------------------------------------------------------------

class TFIDFBlocker:
    """Word n-gram TF-IDF blocker.
    
    Key design choices:
    - analyzer='word', ngram_range=(1,2): word unigrams + bigrams are far more
      discriminative than char 3-grams for business name matching at 5M scale.
    - Vocabulary fitted on S1 + target sample: ensures query-side terms are
      in-vocabulary; fitting only on target caused ~random recall.
    - sublinear_tf=True: dampens high-frequency terms (e.g. 'solutions').
    - GPU via CuPy cuSPARSE sparse matmul; chunked target to fit in VRAM.
    """

    def __init__(self, top_k: int = 50, device: str = "cuda:0",
                 max_features: int = 300_000):
        self.top_k = top_k
        self.device = device
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            dtype=np.float32,
            sublinear_tf=True,
        )

    def fit_transform_target(
        self,
        target_texts: List[str],
        query_texts: List[str],           # S1 texts — included in vocab fitting
        vocab_sample_size: int = 200_000,
        batch_size: int = 500_000,
    ) -> csr_matrix:
        """Fit vocab on S1 + target sample, then transform all target records."""
        import random

        # --- Build vocabulary from S1 + sample of target ---
        target_sample_size = max(0, vocab_sample_size - len(query_texts))
        if len(target_texts) > target_sample_size:
            sample_idx = random.sample(range(len(target_texts)), target_sample_size)
            target_sample = [target_texts[i] for i in sample_idx]
        else:
            target_sample = target_texts

        fit_texts = list(query_texts) + target_sample
        print(f"  Fitting vocab on {len(fit_texts):,} records "
              f"({len(query_texts):,} S1 + {len(target_sample):,} target sample)...", flush=True)
        self.vectorizer.fit(fit_texts)
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_):,} features", flush=True)

        # --- Transform all target records in batches ---
        n_batches = (len(target_texts) + batch_size - 1) // batch_size
        print(f"  Transforming {len(target_texts):,} target records in {n_batches} batches...", flush=True)
        mats = []
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, len(target_texts))
            t0 = time.time()
            mats.append(self.vectorizer.transform(target_texts[s:e]))
            print(f"    Batch {b+1}/{n_batches} ({s:,}..{e:,}) done in {time.time()-t0:.1f}s", flush=True)

        print("  Stacking batches...", flush=True)
        return csr_matrix(sp.vstack(mats, format="csr"))

    def retrieve_candidates(
        self,
        s1_texts: List[str],
        target_matrix: csr_matrix,
        top_k: Optional[int] = None,
        target_chunk_size: int = 200_000,
    ) -> List[List[int]]:
        """Retrieve top-K candidates using CuPy GPU sparse matmul (CPU fallback).

        Chunks the target matrix so result matrices stay small:
          GPU: (n_queries × chunk_size) dense float32 — fits in T4 VRAM.
        """
        k = top_k or self.top_k
        n_queries = len(s1_texts)
        n_targets = target_matrix.shape[0]
        n_chunks = (n_targets + target_chunk_size - 1) // target_chunk_size

        print(f"  Transforming {n_queries:,} S1 query records...", flush=True)
        s1_matrix = self.vectorizer.transform(s1_texts)

        cp, csp = _try_import_cupy()
        use_gpu = (cp is not None) and torch.cuda.is_available()
        device_label = self.device if use_gpu else "CPU"

        print(f"  Chunked top-{k} retrieval on {device_label}: "
              f"{n_queries:,} queries × {n_targets:,} targets "
              f"in {n_chunks} chunks of {target_chunk_size:,}...", flush=True)

        if use_gpu:
            try:
                s1_gpu = csp.csr_matrix(s1_matrix.astype(np.float32))
                print(f"  ✅ S1 uploaded to GPU "
                      f"({s1_gpu.nnz:,} nnz, {s1_gpu.data.nbytes/1e6:.1f} MB)", flush=True)
            except Exception as ex:
                print(f"  ⚠️ GPU upload failed ({ex}), using CPU.", flush=True)
                use_gpu = False

        top_scores = np.full((n_queries, k), -1.0, dtype=np.float32)
        top_indices = np.full((n_queries, k), -1, dtype=np.int32)
        total_t0 = time.time()

        for c_idx in range(n_chunks):
            c_start = c_idx * target_chunk_size
            c_end = min(c_start + target_chunk_size, n_targets)
            chunk = target_matrix[c_start:c_end]

            t0 = time.time()
            if use_gpu:
                try:
                    chunk_gpu = csp.csr_matrix(chunk.astype(np.float32))
                    sim_gpu = (s1_gpu @ chunk_gpu.T).toarray()
                    sim = cp.asnumpy(sim_gpu)
                    del sim_gpu, chunk_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    label = "GPU"
                except Exception as ex:
                    print(f"  ⚠️ GPU chunk failed ({ex}), CPU fallback.", flush=True)
                    sim = s1_matrix.dot(chunk.T).toarray()
                    label = "CPU"
            else:
                sim = s1_matrix.dot(chunk.T).toarray()
                label = "CPU"

            local_k = min(k, sim.shape[1])
            local_best = np.argpartition(sim, -local_k, axis=1)[:, -local_k:]
            local_scores = sim[np.arange(n_queries)[:, None], local_best]
            local_idx = (local_best + c_start).astype(np.int32)

            top_scores, top_indices = _merge_topk(
                top_scores, top_indices, local_scores, local_idx, k)

            elapsed = time.time() - total_t0
            print(f"    [{label}] Chunk {c_idx+1}/{n_chunks} "
                  f"({c_start:,}..{c_end:,}) in {time.time()-t0:.1f}s "
                  f"[{elapsed:.0f}s elapsed]", flush=True)

        print(f"  ✅ Done in {time.time()-total_t0:.1f}s total.", flush=True)
        return [top_indices[i][top_indices[i] >= 0].tolist() for i in range(n_queries)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k_per_source: int = 50,
) -> pd.DataFrame:
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
    print(f"\n── S2 blocking on {dev_s2} ──", flush=True)
    mat_s2 = blocker_s2.fit_transform_target(s2_texts, query_texts=s1_texts)
    cands_s2 = blocker_s2.retrieve_candidates(s1_texts, mat_s2)

    blocker_s3 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s3)
    print(f"\n── S3 blocking on {dev_s3} ──", flush=True)
    mat_s3 = blocker_s3.fit_transform_target(s3_texts, query_texts=s1_texts)
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
