# Amazon ML Entity Resolution Challenge

Entity resolution pipeline for matching records across multiple sources, evaluated using the $F_{0.5}$ metric.

## 📁 Repository Structure

```text
amazon-ml-entity-resolution/
│
├── dataset/                    # Local dataset (git-ignored)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   │
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/                        # Modular source code
│   ├── data.py                 # Data loading utilities
│   ├── normalize.py            # String cleaning & normalization
│   ├── blocking.py             # Blocking & candidate pair generation
│   ├── features.py             # Similarity feature computation
│   ├── model.py                # Pairwise matching ML model
│   ├── evaluate.py             # F0.5 evaluation & metrics
│   └── pipeline.py             # End-to-end execution pipeline
│
├── notebooks/                  # Exploratory Jupyter Notebooks
│   ├── 01_eda.ipynb
│   ├── 02_blocking.ipynb
│   └── 03_model.ipynb
│
├── output/                     # Generated TSV outputs & candidates
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
│
├── Documentation_template.md   # Solution methodology documentation
├── README.md                   # Project overview & guidelines
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

## 👥 Team Workflows & Branch Breakdown

- **Person 1 (`person1-blocking`)**: Data exploration, normalization, blocking, and candidate generation (`normalize.py`, `blocking.py`).
- **Person 2 (`person2-model`)**: Feature engineering, training pair generation, model selection, threshold tuning (`features.py`, `model.py`).
- **Person 3 (`person3-evaluation`)**: $F_{0.5}$ evaluation metric, validation setup, experiment tracking, pipeline orchestration (`evaluate.py`, `pipeline.py`).

## 🚀 Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Dataset Placement:**
   Place the competition dataset inside `dataset/train/` and `dataset/test/`. Note that `dataset/` is git-ignored to prevent committing competition data.