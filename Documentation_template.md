# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** DataResolvers  
**Team Members:** Girish Kumar Yadav  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed a two-stage entity resolution pipeline combining a high-recall dual-pass TF-IDF blocking stage (achieving **96.53% blocking recall**) with a precision-optimised pairwise binary classifier achieving a **cross-validation Macro F0.5 of 0.8664** on the training sample. Our key innovations are an Indic-language-safe text normalizer shared across both stages, a dual-pass word + character n-gram blocking union that recovers typos and cross-script business name variations, and a 15-feature similarity engine combining fuzzy string metrics, phonetic Double Metaphone matching, and address building number matching.

---

## 2. Methodology

### 2.1 Problem Analysis

Key insights from EDA on the 2.2M + 5M + 5.3M source datasets:

- **Language Diversity**: ~30% of records contain Indic scripts (Devanagari/Tamil). Standard NFKD Unicode normalization shreds Indic vowel signs requiring script-aware folding.
- **DBA Markers**: ~8% of records contain `d/b/a`, `D.B.A.`, `doing business as` markers that must be stripped before punctuation removal.
- **Legal Suffix Noise**: `Inc`, `LLC`, `Corp`, `Pvt Ltd`, `S.A.R.L.` appear inconsistently and must be stripped before similarity computation.
- **Shared Addresses**: Indic-name records often share Latin addresses — including address text in blocking was essential.
- **Class Imbalance**: ~1:30 positive-to-negative ratio in candidate pairs, handled via `class_weight="balanced"`.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Pairwise Binary Classifier  
**Core Innovation:** Dual-pass TF-IDF blocking union (word + character n-gram) with Indic-safe normalization, achieving 96.53% blocking recall on 5M+ target records.

---

## 3. Candidate Generation (Blocking)

**Blocking method:** Dual-pass TF-IDF cosine similarity with GPU-accelerated top-k retrieval.

| Pass | Analyzer | N-gram | Max features | GPU | Purpose |
|------|----------|--------|-------------|-----|---------|
| Pass 1 | Word | (1,2) | 150,000 | CUDA | Exact/near-exact name matches |
| Pass 2 | char_wb | (2,4) | 200,000 | CPU fallback | Typos, domains, partial overlaps |

- **Text representation:** `business_name + " " + business_address`
- **Candidate pairs generated:** 521,632 for 5,000 S1 queries (~104.3 per entity)
- **Blocking Recall:** **96.53%** (16,760 / 17,362 ground truth matches retrieved)
- **True match preservation:** Character n-gram union recovers typos and cross-script overlaps; address text recovers Indic-name pairs sharing Latin addresses.

---

## 4. Matching Model

**Features used (15 total):**

| Feature | Description |
|---------|-------------|
| `name_jaccard` | Token Jaccard on raw names |
| `name_levenshtein` | Edit distance ratio on normalized names |
| `name_partial_ratio` | RapidFuzz partial string match |
| `name_token_sort` | RapidFuzz token sort ratio |
| `name_token_set` | RapidFuzz token set ratio |
| `name_exact_match` | Binary: identical normalized names |
| `name_first_token_match` | Binary: lead brand word matches |
| `phonetic_match` | Double Metaphone phonetic code overlap |
| `addr_jaccard` | Token Jaccard on addresses |
| `addr_levenshtein` | Edit distance on normalized addresses |
| `addr_partial_ratio` | Partial match on addresses |
| `addr_number_match` | Building/street number Jaccard |
| `same_country` | Binary country equality |
| `name_len_ratio` | Normalized name length ratio |
| `addr_len_ratio` | Normalized address length ratio |

**Model type:** Random Forest (`n_estimators=200, max_depth=12, class_weight="balanced"`) + LightGBM (`n_estimators=300, scale_pos_weight=5.0`)  
**Threshold selection:** Sweep precision-recall thresholds; maximize F0.5 on each CV fold. Optimal threshold: **0.9679**.

---

## 5. Results & Error Analysis

| Metric | Value |
|--------|-------|
| Blocking Recall | **96.53%** |
| CV Macro Precision | **92.83%** |
| CV Macro Recall | **68.41%** |
| **CV Macro F0.5** | **0.8664** |
| Feature extraction speed | 7,249 pairs/sec |

**Common false positives:** Different businesses at the same address (same-building tenants) score high on `addr_jaccard`.  
**Common false negatives (blocking):** Cross-script pairs with no shared address or Latin character overlap.  
**Common false negatives (model):** Heavily abbreviated names (`"N.O. PLLC"` vs `"Novent Owl PLLC"`) — low string similarity despite high phonetic similarity.

---

## 6. Conclusion

We achieved **96.53% blocking recall** via dual-pass TF-IDF (word + character n-gram union) with name+address text and Indic-safe normalization. The 15-feature pairwise Random Forest classifier achieved **CV Macro F0.5 of 0.8664** with 92.83% precision at optimal threshold. The key lesson: text normalization correctness is foundational — early Unicode and DBA-marker bugs caused measurable recall loss in both blocking and feature stages.

---

## Appendix

### A. Code Artefacts

Code ships under `code/business_entity_resolution/src/`. Entry points:

```bash
# Validation run:
python src/pipeline.py --mode val --data-dir dataset/val_sample

# Test inference:
python src/pipeline.py --mode test --data-dir dataset/test --threshold 0.9679
```

**Recall progression:**

| Version | Recall |
|---------|--------|
| Baseline word n-gram (name only) | 59.50% |
| + Address text | ~64% |
| + `max_df=0.5` | ~65% |
| + Char (2,4)-gram union | **96.53%** |

### B. Additional Results

Top feature importances (Random Forest on 521,632 pairs):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `addr_jaccard` | 18.44% |
| 2 | `name_token_set` | 15.37% |
| 3 | `name_jaccard` | 14.59% |
| 4 | `name_partial_ratio` | 12.66% |
| 5 | `name_levenshtein` | 9.74% |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.


---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 6. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
