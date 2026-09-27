"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.

Key design decisions:
- Dual-pass TF-IDF: word (1,2)-gram + char (2,4)-gram union for max recall
- Word pass: catches exact/near-exact business name matches
- Char pass: catches typos, glued domains, partial word overlap
- Name + address text: captures Indic-name pairs sharing Latin addresses
- max_df=0.5: filters ultra-common tokens → sparser matmul → faster chunks
- Vocab fitted on S1 + target sample: prevents S1 terms being OOV
- Sequential dual-GPU (default): S2 then S3 to stay under Kaggle 13GB RAM
"""

import time
import threading
import queue
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

def _normalize_text_fast(text: str) -> str:
    """Lowercase + accent strip + remove legal suffixes + punctuation cleanup."""
    import unicodedata, re
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(
        r'\b(llc|ltd|inc|corp|co|plc|gmbh|llp|lp|sa|srl|sl|bv|nv|ag|oy|ab|as|pte|pvt|sas|kk|kg)\b\.?',
        '', text
    )
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_record_texts(df: pd.DataFrame) -> List[str]:
    """Combine business name + address into a single normalized string.

    Including address captures Indic-name pairs that share Latin addresses
    (e.g. Tamil S2 name + '6(29), C.I.T. COLONY' matches Latin S1 with same address).
    """
    names = df["business_name"].fillna("").astype(str)
    addrs = df["business_address"].fillna("").astype(str)
    combined = (names + " " + addrs).apply(_normalize_text_fast)
    return combined.tolist()


# ---------------------------------------------------------------------------
# CuPy helper
# ---------------------------------------------------------------------------

def _try_import_cupy():
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        cp.array([1.0])
        return cp, csp
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Sparse top-k extraction (NO .toarray() — zero peak-RAM overhead)
# ---------------------------------------------------------------------------

def _topk_from_sparse(sim_sparse: csr_matrix, k: int, col_offset: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract top-k (score, global_col_index) per row from a CSR sparse matrix.

    Works directly on indptr/indices/data — never allocates a dense matrix.
    Returns arrays of shape (n_rows, min(k, max_nnz_per_row)).
    """
    n_rows = sim_sparse.shape[0]
    indptr = sim_sparse.indptr
    indices = sim_sparse.indices
    data = sim_sparse.data

    row_scores = []
    row_cols = []
    for i in range(n_rows):
        s, e = indptr[i], indptr[i + 1]
        n = e - s
        if n == 0:
            row_scores.append(np.empty(0, dtype=np.float32))
            row_cols.append(np.empty(0, dtype=np.int32))
            continue
        r_data = data[s:e]
        r_ind = indices[s:e]
        local_k = min(k, n)
        if n <= local_k:
            order = np.argsort(r_data)[::-1]
        else:
            top_pos = np.argpartition(r_data, -local_k)[-local_k:]
            order = top_pos[np.argsort(r_data[top_pos])[::-1]]
        row_scores.append(r_data[order].astype(np.float32))
        row_cols.append((r_ind[order] + col_offset).astype(np.int32))

    return row_scores, row_cols


def _merge_topk_lists(
    running_scores: List[np.ndarray],
    running_idx: List[np.ndarray],
    new_scores: List[np.ndarray],
    new_idx: List[np.ndarray],
    k: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Merge per-row running top-k with new chunk top-k (list-of-array format)."""
    merged_s, merged_i = [], []
    for rs, ri, ns, ni in zip(running_scores, running_idx, new_scores, new_idx):
        cs = np.concatenate([rs, ns])
        ci = np.concatenate([ri, ni])
        if len(cs) <= k:
            merged_s.append(cs)
            merged_i.append(ci)
        else:
            best = np.argpartition(cs, -k)[-k:]
            best = best[np.argsort(cs[best])[::-1]]
            merged_s.append(cs[best])
            merged_i.append(ci[best])
    return merged_s, merged_i


# ---------------------------------------------------------------------------
# TFIDFBlocker
# ---------------------------------------------------------------------------

class TFIDFBlocker:
    """TF-IDF blocker with configurable analyzer, GPU sparse matmul, and zero-copy top-k."""

    def __init__(self, top_k: int = 50, device: str = "cuda:0",
                 max_features: int = 150_000,
                 analyzer: str = "word", ngram_range: Tuple[int, int] = (1, 2),
                 max_df: float = 0.5):
        self.top_k = top_k
        self.device = device
        self.device_id = int(device.split(":")[-1]) if "cuda:" in device else 0
        self.analyzer = analyzer
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=2,
            max_df=max_df,
            max_features=max_features,
            dtype=np.float32,
            sublinear_tf=True,
        )

    def fit_transform_target(
        self,
        target_texts: List[str],
        query_texts: List[str],
        vocab_sample_size: int = 200_000,
        batch_size: int = 500_000,
    ) -> csr_matrix:
        """Fit vocab on S1+target combined, transform all target records in batches."""
        target_sample_n = max(0, vocab_sample_size - len(query_texts))
        rng = np.random.default_rng(42)
        target_sample = (
            [target_texts[i] for i in rng.choice(len(target_texts), size=target_sample_n, replace=False)]
            if len(target_texts) > target_sample_n else target_texts
        )
        fit_texts = list(query_texts) + target_sample
        print(f"  Fitting vocab on {len(fit_texts):,} texts "
              f"({len(query_texts):,} S1 + {len(target_sample):,} target sample)...", flush=True)
        self.vectorizer.fit(fit_texts)
        print(f"  Vocab: {len(self.vectorizer.vocabulary_):,} features", flush=True)

        n_batches = (len(target_texts) + batch_size - 1) // batch_size
        print(f"  Transforming {len(target_texts):,} records in {n_batches} batches...", flush=True)
        mats = []
        for b in range(n_batches):
            s, e = b * batch_size, min((b + 1) * batch_size, len(target_texts))
            t0 = time.time()
            mats.append(self.vectorizer.transform(target_texts[s:e]))
            print(f"    Batch {b+1}/{n_batches} ({s:,}..{e:,}) in {time.time()-t0:.1f}s", flush=True)

        print("  Stacking batches...", flush=True)
        return csr_matrix(sp.vstack(mats, format="csr"))

    def retrieve_candidates(
        self,
        s1_texts: List[str],
        target_matrix: csr_matrix,
        top_k: Optional[int] = None,
        target_chunk_size: int = 100_000,
        gpu_query_batch: int = 500,
    ) -> List[List[int]]:
        """Retrieve top-K candidates.

        CPU path: sparse dot product → top-k extracted from sparse CSR rows (no dense alloc).
        GPU path: dense similarity kept on GPU; only per-query top-k values+indices cross
        PCIe (~50KB/batch). Zero-score candidates (empty/zero-vector S1 rows) are filtered.
        """
        k = top_k or self.top_k
        n_queries = len(s1_texts)
        n_targets = target_matrix.shape[0]
        n_chunks = (n_targets + target_chunk_size - 1) // target_chunk_size

        print(f"  Transforming {n_queries:,} S1 queries...", flush=True)
        s1_matrix = self.vectorizer.transform(s1_texts)

        cp, csp = _try_import_cupy()
        use_gpu = cp is not None and torch.cuda.is_available()
        device_label = self.device if use_gpu else "CPU"

        print(f"  Top-{k} retrieval on {device_label}: "
              f"{n_queries:,}×{n_targets:,} in {n_chunks} chunks of {target_chunk_size:,}...",
              flush=True)

        if use_gpu:
            try:
                with cp.cuda.Device(self.device_id):
                    s1_gpu = csp.csr_matrix(s1_matrix.astype(np.float32))
                    print(f"  ✅ S1 on {self.device} ({s1_gpu.nnz:,} nnz, {s1_gpu.data.nbytes/1e6:.1f} MB)", flush=True)
            except Exception as ex:
                print(f"  ⚠️ GPU upload failed: {ex} — using CPU.", flush=True)
                use_gpu = False

        # Initialise running top-k as lists of empty arrays
        running_scores = [np.empty(0, np.float32) for _ in range(n_queries)]
        running_idx    = [np.empty(0, np.int32)   for _ in range(n_queries)]
        total_t0 = time.time()

        for c_idx in range(n_chunks):
            c_start = c_idx * target_chunk_size
            c_end   = min(c_start + target_chunk_size, n_targets)
            chunk   = target_matrix[c_start:c_end]   # (chunk_size × vocab) sparse
            t0 = time.time()

            if use_gpu:
                try:
                    with cp.cuda.Device(self.device_id):
                        chunk_gpu = csp.csr_matrix(chunk.astype(np.float32))
                        # Batch queries to keep dense result small: gpu_query_batch × chunk_size × 4B
                        chunk_scores, chunk_cols = [], []
                        for qb in range(0, n_queries, gpu_query_batch):
                            s1_sub_gpu = s1_gpu[qb: qb + gpu_query_batch]
                            # GPU sparse matmul → transfer dense to CPU → numpy argpartition
                            # (numpy argpartition is 28x faster than cupy's for 500×100k arrays)
                            sim_dense = cp.asnumpy((s1_sub_gpu @ chunk_gpu.T).toarray())
                            local_k = min(k, sim_dense.shape[1])
                            bpos = np.argpartition(sim_dense, -local_k, axis=1)[:, -local_k:]
                            bscores = sim_dense[np.arange(sim_dense.shape[0])[:, None], bpos]
                            bcols = (bpos + c_start).astype(np.int32)
                            # Filter zero-score candidates (empty/zero-vector S1 rows
                            # get arbitrary indices from argpartition — pure noise)
                            for r in range(bscores.shape[0]):
                                mask = bscores[r] > 0
                                chunk_scores.append(bscores[r][mask].astype(np.float32))
                                chunk_cols.append(bcols[r][mask].astype(np.int32))
                        del chunk_gpu
                        cp.get_default_memory_pool().free_all_blocks()
                        label = f"GPU ({self.device})"
                except Exception as ex:
                    print(f"  ⚠️ GPU chunk failed: {ex} — CPU fallback.", flush=True)
                    # Clean up any GPU memory from partial execution
                    try:
                        del chunk_gpu
                    except NameError:
                        pass
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass
                    sim_sparse = s1_matrix.dot(chunk.T).tocsr()
                    chunk_scores, chunk_cols = _topk_from_sparse(sim_sparse, k, c_start)
                    label = "CPU"
            else:
                # Sparse dot → sparse result → zero-copy top-k extraction
                sim_sparse = s1_matrix.dot(chunk.T).tocsr()
                chunk_scores, chunk_cols = _topk_from_sparse(sim_sparse, k, c_start)
                label = "CPU"

            running_scores, running_idx = _merge_topk_lists(
                running_scores, running_idx, chunk_scores, chunk_cols, k)

            print(f"    [{label}] Chunk {c_idx+1}/{n_chunks} "
                  f"({c_start:,}..{c_end:,}) in {time.time()-t0:.1f}s "
                  f"[{time.time()-total_t0:.0f}s total]", flush=True)

        if use_gpu:
            try:
                with cp.cuda.Device(self.device_id):
                    del s1_gpu
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        print(f"  ✅ Done in {time.time()-total_t0:.1f}s.", flush=True)
        return [idx.tolist() for idx in running_idx]


# ---------------------------------------------------------------------------
# Main entry point — sequential dual-GPU execution (RAM safe)
# ---------------------------------------------------------------------------

def _blocking_worker(
    name: str,
    s1_texts: List[str],
    target_texts: List[str],
    top_k: int,
    device: str,
    result_q: queue.Queue,
) -> None:
    """Worker function for threaded dual-GPU blocking."""
    try:
        blocker = TFIDFBlocker(top_k=top_k, device=device)
        mat = blocker.fit_transform_target(target_texts, query_texts=s1_texts)
        cands = blocker.retrieve_candidates(s1_texts, mat)
        result_q.put((name, cands, None))
    except Exception as ex:
        result_q.put((name, None, ex))


def _run_single_pass(
    pass_name: str,
    s1_texts: List[str],
    target_texts: List[str],
    top_k: int,
    device: str,
    analyzer: str = "word",
    ngram_range: Tuple[int, int] = (1, 2),
    max_df: float = 0.5,
    max_features: int = 150_000,
) -> List[List[int]]:
    """Run a single TF-IDF blocking pass with the given analyzer config."""
    import gc
    print(f"\n  📌 {pass_name} pass (analyzer={analyzer}, ngram={ngram_range}, max_df={max_df})...",
          flush=True)
    blocker = TFIDFBlocker(
        top_k=top_k, device=device,
        max_features=max_features,
        analyzer=analyzer, ngram_range=ngram_range, max_df=max_df,
    )
    mat = blocker.fit_transform_target(target_texts, query_texts=s1_texts)
    cands = blocker.retrieve_candidates(s1_texts, mat)
    del mat, blocker
    gc.collect()
    return cands


def _merge_candidate_indices(
    cands_a: List[List[int]],
    cands_b: List[List[int]],
) -> List[List[int]]:
    """Union two candidate index lists (per-query)."""
    merged = []
    for a, b in zip(cands_a, cands_b):
        merged.append(list(set(a) | set(b)))
    return merged


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k_per_source: int = 30,
) -> pd.DataFrame:
    """Generate candidate entity pairs using single-pass word TF-IDF blocking.

    Uses name + address text to capture Indic-name pairs sharing Latin addresses.
    max_df=0.5 filters ultra-common tokens for better discrimination.
    Sequential execution: S2 on cuda:0, then S3 on cuda:1 to stay under Kaggle RAM.
    """
    import gc
    print("Preparing record text representations (name + address)...", flush=True)
    s1_texts = get_record_texts(df_s1)
    s2_texts = get_record_texts(df_s2)
    s3_texts = get_record_texts(df_s3)

    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    dev_s2 = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_s3 = "cuda:1" if torch.cuda.device_count() > 1 else dev_s2

    # ── S2 blocking ──
    print(f"\n── S2 blocking on {dev_s2} ──", flush=True)
    cands_s2 = _run_single_pass(
        "Word n-gram", s1_texts, s2_texts,
        top_k=top_k_per_source, device=dev_s2,
        analyzer="word", ngram_range=(1, 2), max_df=0.5, max_features=150_000,
    )
    gc.collect()

    # ── S3 blocking ──
    print(f"\n── S3 blocking on {dev_s3} ──", flush=True)
    cands_s3 = _run_single_pass(
        "Word n-gram", s1_texts, s3_texts,
        top_k=top_k_per_source, device=dev_s3,
        analyzer="word", ngram_range=(1, 2), max_df=0.5, max_features=150_000,
    )
    gc.collect()

    # ── Format results ──
    print("\nFormatting candidate results...", flush=True)
    results = []
    total_cands = 0
    for idx, s1_id in enumerate(df_s1["entity_id"].tolist()):
        matched = set()
        for s2_idx in cands_s2[idx]:
            matched.add(s2_ids[s2_idx])
        for s3_idx in cands_s3[idx]:
            matched.add(s3_ids[s3_idx])
        total_cands += len(matched)
        results.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(sorted(matched))
        })

    n_queries = len(df_s1)
    print(f"✅ {total_cands:,} total candidates for {n_queries:,} queries "
          f"(avg {total_cands/n_queries:.1f} per query)", flush=True)
    return pd.DataFrame(results)
