# Amazon ML Entity Resolution Challenge

Entity resolution pipeline for matching business records across multiple disparate sources (Source 1, Source 2, Source 3) evaluated using the competition's official precision-weighted **Macro $F_{0.5}$** metric.

---

## 📁 Repository Structure

```text
amazon-ml-entity-resolution/
│
├── dataset/                    # Local dataset (git-ignored / linked)
│   ├── train/                  # Full training data & ground truth
│   ├── val_sample/             # Stratified 9,999-entity validation benchmark
│   └── test/                   # Competition test set (1.73M entities)
│
├── src/                        # Modular source code
│   ├── data.py                 # Fast TSV I/O & ground truth mapping
│   ├── normalize.py            # String cleaning, accents, legal suffixes, DBAs
│   ├── blocking.py             # Inverted index blocking with bucket frequency pruning
│   ├── features.py             # 11-dimensional similarity feature engineering (RapidFuzz)
│   ├── model.py                # Pairwise matching classifier (Balanced Random Forest)
│   ├── evaluate.py             # Official Macro F0.5 evaluation engine & singleton metrics
│   ├── threshold.py            # Decision threshold optimization [0.10, 0.95]
│   ├── tracker.py              # Experiment tracking logging to experiments/experiments.csv
│   ├── error_analysis.py       # Automated FP/FN diagnosis (blocking vs model loss)
│   ├── validate_submission.py  # Wrapper for official competition submission validator
│   └── pipeline.py             # End-to-end orchestrated validation & test pipeline
│
├── outputs/                    # Validation outputs & saved model artifacts
│   ├── model.joblib            # Trained Random Forest matching model
│   └── val_run/                # Validation predictions & candidate pairs
│
├── output/                     # Official submission deliverables
│   ├── matching_results.tsv    # Predicted matches (source1_entity_id -> matched_entity_ids)
│   └── candidate_pairs.tsv     # Candidate pairs superset (source1_entity_id -> candidate_entity_ids)
│
├── experiments/                # Experiment tracking history
│   └── experiments.csv         # Logged validation runs, models, thresholds, & scores
│
├── Documentation_template.md   # Official solution methodology documentation
├── README.md                   # Project overview & guidelines
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

---

## 👥 Team Workflows & Responsibilities

- **Person 1 (`person1-blocking`)**: Text normalization, legal suffix extraction, inverted index blocking, candidate generation, and candidate recall maximization.
- **Person 2 (`person2-model`)**: Feature engineering (11 similarity dimensions), hard negative training pair sampling, balanced classifier training, and feature importance analysis.
- **Person 3 (`person3-evaluation`)**: Official Macro $F_{0.5}$ metric engine, singleton analysis, decision threshold optimization, experiment tracking, end-to-end pipeline orchestration, and submission validation.

---

## 📊 Benchmark Results (Validation Set)

Evaluated on the representative 9,999-entity validation sample (`dataset/val_sample`):

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Official Leaderboard Metric: Macro $F_{0.5}$** | **0.5406** | Precision-weighted 2× over recall |
| **Macro Precision** | **0.6300** | High precision prevents catastrophic singleton penalties |
| **Macro Recall** | **0.4184** | Captures 97.7% of candidates shortlisted by blocking |
| **Singleton Identification Accuracy** | **95.8%** | 530 / 553 zero-match singletons identified correctly |
| **Optimal Classification Threshold** | **0.45** | Empirically selected via threshold sweep |
| **Blocking Candidate Recall** | **0.3980** | 13,764 / 34,581 true matches retrieved in candidate pool |

### Breakdown by Country
- **United States (US)**: Macro $F_{0.5} = 0.5470$ (Precision: 0.6374, Recall: 0.4225)
- **India**: Macro $F_{0.5} = 0.5309$ (Precision: 0.6187, Recall: 0.4122)

---

## 🚀 Reproduction & Execution Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Validation Pipeline
Executes blocking, extracts 11 features, fits the model, sweeps thresholds, outputs evaluation report, and saves `outputs/model.joblib`:
```bash
python src/pipeline.py --mode val
```

### 3. Run Standalone Metric Evaluation
```bash
python src/evaluate.py --predictions outputs/val_run/matching_results.tsv --ground-truth dataset/val_sample/val_ground_truth.tsv
```

### 4. Run Competition Test Inference
Generates the official submission files in `output/` with streaming country-partitioned batching:
```bash
python src/pipeline.py --mode test --threshold 0.45
```

### 5. Validate Submission Files
Checks syntax, headers, row counts, singleton formatting, and superset constraints:
```bash
python src/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```