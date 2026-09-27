# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DataResolvers  
**Team Members:** Shray Singal & Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We present an end-to-end, high-performance Business Entity Resolution pipeline designed to accurately match enterprise records across three disparate sources (Source 1, Source 2, Source 3) under the precision-heavy **Macro $F_{0.5}$** evaluation metric. Our architecture unites a country-partitioned inverted index blocker with frequency bucket pruning, an 11-dimensional lexical and phonetic similarity feature space, and a class-balanced Random Forest classifier with empirical decision threshold optimization. By implementing streaming country batching and candidate-focused normalization caching, the entire solution resolves 1.73 million test entities across ~10 million candidate records on standard hardware in $<1.5$ GB RAM without requiring external APIs or GPU acceleration.

---

## 2. Methodology

### 2.1 Problem Analysis
Key insights discovered during exploratory data analysis and validation profiling:
1. **Multi-Country Noise Patterns**: The dataset covers three distinct geographic jurisdictions: India, the United States, and France. Each exhibits distinct structural variations (e.g., French street articles, Indian state/honorific prefixes such as "Mr", "Smt", "M/S", and US corporate entity abbreviations).
2. **High Singleton Prevalence**: A substantial portion of Source 1 entities have zero true matches in Source 2 or Source 3. Under Macro $F_{0.5}$, predicting even a single false positive for a true singleton scores 0.0 for that entity, heavily penalizing over-merging.
3. **Blocking Recall Ceiling**: Quantitative error analysis revealed that 98% of baseline false negatives stemmed from candidate generation misses rather than model rejections. Business names frequently vary by word transposition, legal suffixes (`Pvt Ltd`, `LLC`, `Corp`), or DBA ("doing business as") prefixes, while sharing valid addresses.

### 2.2 Solution Strategy
We structured the solution into a decoupled, highly modular three-stage architecture:

**Approach Type:** Country-Partitioned Inverted Index Blocking + Pairwise Gradient Tree/Random Forest Classifier + $F_{0.5}$ Threshold Sweeping  
**Core Innovation:** A memory-safe streaming inference engine with dual-tier candidate shortlisting. Rather than generating a full Cartesian product or storing tens of millions of feature vectors in memory, our pipeline streams candidate scoring in batches of 100,000 pairs with pre-tokenized C-accelerated set intersections, achieving a throughput of >40,000 pairs/sec with $<1.5$ GB peak RAM consumption.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 10\text{M} \approx 1.7 \times 10^{13}$ comparison space into a high-recall candidate pool:

- **Blocking keys used:**
  1. `Country + Prefix-3`: First 3 alphanumeric characters of normalized business name.
  2. `Country + First Token`: First whitespace-delimited word of normalized business name.
  3. `Country + Longest Distinctive Token`: The longest token in the name with $\ge 4$ characters.
- **Frequency Bucket Pruning**: Inverted index buckets containing $>500$ entities (e.g., generic industry words like "store", "enterprises", "trading") are pruned to eliminate combinatorial explosion and stop-word noise.
- **Candidate Cap**: Maximum of 150 candidates per Source 1 entity, ensuring strict upper bounds on pairwise scoring.
- **Candidate pairs generated:** ~14.6M for France, ~40M for India, and ~32M for US.
- **Preserving True Matches**: Country-strict partitioning guarantees zero cross-country candidate pollution. Combining first-token and longest-token keys successfully captures transposed names (e.g., "Sharma Medical Store" matching "Medical Store Sharma").

---

## 4. Matching Model

### Features Used (11 Dimensions)
- **Business Name Features**:
  1. `name_jaccard`: Fast word-level token set Jaccard similarity.
  2. `name_levenshtein`: Normalized Levenshtein ratio via RapidFuzz.
  3. `name_partial_ratio`: Partial substring alignment score.
  4. `name_token_sort`: Token-sorted string similarity (robust to word order changes).
  5. `name_token_set`: Token set ratio (robust to subset/superset business names).
  6. `name_len_ratio`: Normalized character length ratio $\frac{\min(|a|, |b|)}{\max(|a|, |b|)}$.
- **Business Address Features**:
  7. `addr_jaccard`: Fast word-level address token set Jaccard similarity.
  8. `addr_levenshtein`: Normalized address Levenshtein ratio.
  9. `addr_partial_ratio`: Substring alignment score for addresses (captures building/street overlap).
  10. `addr_len_ratio`: Normalized character length ratio of addresses.
- **Metadata Consistency**:
  11. `same_country`: Binary indicator verifying country equality.

### Model Architecture
- **Model Type**: Balanced Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`, `n_jobs=-1`).
- **Feature Importance**: Address Jaccard (0.2956) and Address Partial Ratio (0.2890) proved to be the strongest discriminators, followed by Name Jaccard (0.1634) and Name Levenshtein (0.1587).
- **Threshold Selection Method**: Exhaustive sweep over thresholds $\theta \in [0.10, 0.95]$ with step 0.05 on the stratified validation set to maximize the competition metric:
  $$\text{Macro } F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Optimal threshold identified: **$\theta = 0.45$**.

---

## 5. Results & Error Analysis

### Official Validation Benchmark (9,999 S1 Entities, 34,581 Actual Matches)
- **Macro $F_{0.5}$ Score**: **0.5406**
- **Macro Precision**: **0.6300**
- **Macro Recall**: **0.4184**
- **Singleton Accuracy**: **95.8%** (530 of 553 true singletons correctly predicted empty)
- **Country Breakdown**:
  - United States: Macro $F_{0.5} = 0.5470$
  - India: Macro $F_{0.5} = 0.5309$

### Error Analysis Diagnostics
- **Common False Positives (Wrong Merges)**:
  - Corporate branch offices and franchise networks sharing identical parent names and city/state tokens but differing by local unit/street number.
- **Common False Negatives (Missed Matches)**:
  - Severe transliteration divergence in non-Latin Indic scripts.
  - Entities with missing or heavily abbreviated addresses where name token overlap was below the candidate generation threshold.

---

## 6. Conclusion

Our solution achieves a strong, robust Macro $F_{0.5}$ score of 0.5406 while adhering to all strict operational constraints. By decoupling the inverted index candidate generator from a streaming, precision-optimized Random Forest scorer, we eliminated memory bottlenecks and achieved complete test inference over 1.73M entities in constant memory on standard hardware.

---

## Appendix

### A. Code Artefacts & Structure
The submission code is structured cleanly under `src/`:
- `src/pipeline.py`: Orchestrates end-to-end validation (`--mode val`) and test inference (`--mode test`).
- `src/blocking.py`: Inverted index construction, blocking keys, and bucket pruning.
- `src/features.py`: C-accelerated 11-dimensional feature computation via RapidFuzz.
- `src/model.py`: Pairwise classifier training, threshold tuning, and probability prediction.
- `src/evaluate.py`: Macro $F_{0.5}$ evaluator with candidate recall and singleton tracking.
- `src/validate_submission.py`: Comprehensive validator ensuring full compliance with competition rules.

**Execution Entry Point to Reproduce Deliverables**:
```bash
# 1. Train model on validation split:
python src/pipeline.py --mode val

# 2. Run full test inference to produce output/matching_results.tsv & output/candidate_pairs.tsv:
python src/pipeline.py --mode test --threshold 0.45
```

