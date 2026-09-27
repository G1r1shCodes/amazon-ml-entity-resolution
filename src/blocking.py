"""Blocking & Candidate Pair Generation Module.

Assigned to: Person 1 (blocking & candidate generation)
Calculates candidate pairs from Source 2 and Source 3 for each Source 1 entity.

Key design decisions:
- Word n-gram TF-IDF (1,2): word token overlap >> char 3-gram at 5M scale
- Vocab fitted on S1 + target sample: prevents S1 terms being OOV
- Sparse-result top-k extraction: NO .toarray() on full result — avoids 4GB RAM spikes
- Sequential dual-GPU (default): S2 on cuda:0, then S3 on cuda:1 — parallel mode exists
  but vectorizing 10M texts concurrently exceeds Kaggle's 13GB RAM (causes kernel restarts)
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

def _normalize_name_fast(name: str) -> str:
    """Lowercase + accent strip + remove legal suffixes for vectorized use."""
    import unicodedata, re
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(
        r'\b(llc|ltd|inc|corp|co|plc|gmbh|llp|lp|sa|srl|sl|bv|nv|ag|oy|ab|as|pte|pvt|sas|kk|kg)\b\.?',
        '', name
    )
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def get_combined_record_strings_vectorized(df: pd.DataFrame) -> List[str]:
    """Normalize business names for TF-IDF blocking (name-only — address is noisy/empty)."""
    return df["business_name"].fillna("").astype(str).apply(_normalize_name_fast).tolist()


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
    """Word n-gram TF-IDF blocker with GPU sparse matmul and zero-copy top-k."""

    def __init__(self, top_k: int = 50, device: str = "cuda:0",
                 max_features: int = 150_000):
        self.top_k = top_k
        self.device = device
        self.device_id = int(device.split(":")[-1]) if "cuda:" in device else 0
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
                            # Dense similarity stays ON GPU — only per-query top-k crosses PCIe
                            # (transfers ~50KB per batch instead of a 200MB dense block)
                            sim_gpu = (s1_sub_gpu @ chunk_gpu.T).toarray()
                            local_k = min(k, sim_gpu.shape[1])
                            bpos_gpu = cp.argpartition(sim_gpu, -local_k, axis=1)[:, -local_k:]
                            bscores_gpu = cp.take_along_axis(sim_gpu, bpos_gpu, axis=1)
                            order_gpu = cp.argsort(-bscores_gpu, axis=1)
                            bpos_gpu = cp.take_along_axis(bpos_gpu, order_gpu, axis=1)
                            bscores_gpu = cp.take_along_axis(bscores_gpu, order_gpu, axis=1)
                            del sim_gpu, order_gpu
                            bscores = cp.asnumpy(bscores_gpu)
                            bcols = cp.asnumpy(bpos_gpu).astype(np.int32) + c_start
                            del bscores_gpu, bpos_gpu
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


def generate_candidate_pairs(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    top_k_per_source: int = 50,
    parallel: bool = False,
) -> pd.DataFrame:
    """Generate candidate entity pairs using TF-IDF blocking.

    Sequential mode (`parallel=False`, default) processes S2 then S3 sequentially.
    When 2 GPUs are available, S2 uses `cuda:0` and S3 uses `cuda:1`.
    Sequential execution ensures RAM usage stays under ~6GB, preventing Kaggle kernel restarts.
    """
    import gc
    print("Preparing record text representations...", flush=True)
    s1_texts = get_combined_record_strings_vectorized(df_s1)
    s2_texts = get_combined_record_strings_vectorized(df_s2)
    s3_texts = get_combined_record_strings_vectorized(df_s3)

    s2_ids = df_s2["entity_id"].tolist()
    s3_ids = df_s3["entity_id"].tolist()

    dev_s2 = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_s3 = "cuda:1" if torch.cuda.device_count() > 1 else dev_s2
    dual_gpu = torch.cuda.device_count() > 1

    if parallel and dual_gpu:
        print(f"\n🔀 Parallel dual-GPU blocking: S2→{dev_s2}, S3→{dev_s3}", flush=True)
        result_q: queue.Queue = queue.Queue()
        t_s2 = threading.Thread(
            target=_blocking_worker,
            args=("s2", s1_texts, s2_texts, top_k_per_source, dev_s2, result_q),
            daemon=True,
        )
        t_s3 = threading.Thread(
            target=_blocking_worker,
            args=("s3", s1_texts, s3_texts, top_k_per_source, dev_s3, result_q),
            daemon=True,
        )
        t_s2.start()
        t_s3.start()
        t_s2.join()
        t_s3.join()

        cands_s2 = cands_s3 = None
        for _ in range(2):
            name, cands, err = result_q.get()
            if err:
                raise RuntimeError(f"Blocking worker '{name}' failed: {err}")
            if name == "s2":
                cands_s2 = cands
            else:
                cands_s3 = cands
    else:
        print(f"\n── Processing S2 blocking on {dev_s2} ──", flush=True)
        blocker_s2 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s2)
        mat_s2 = blocker_s2.fit_transform_target(s2_texts, query_texts=s1_texts)
        cands_s2 = blocker_s2.retrieve_candidates(s1_texts, mat_s2)
        del mat_s2, blocker_s2
        gc.collect()

        print(f"\n── Processing S3 blocking on {dev_s3} ──", flush=True)
        blocker_s3 = TFIDFBlocker(top_k=top_k_per_source, device=dev_s3)
        mat_s3 = blocker_s3.fit_transform_target(s3_texts, query_texts=s1_texts)
        cands_s3 = blocker_s3.retrieve_candidates(s1_texts, mat_s3)
        del mat_s3, blocker_s3
        gc.collect()

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
