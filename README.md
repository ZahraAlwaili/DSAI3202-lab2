# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

**NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001)**

---

## Final Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost |
| **RMSE** | 18.553 |
| **MAE** | 13.117 |
| **R²** | 0.547 |
| **Features Used** | 17 / 140 (87.8% reduction) |
| **Total Runtime** | ~5 minutes |

---

## Pipeline Overview

```
Raw Data (Azure Data Lake — lab5 container)
        ↓
  Raw → Silver (RUL + clean)
        ↓
  Silver → Gold (StandardScaler + RUL Clipping @ 125)
        ↓
  tsfresh Extraction → 140 features (Window=30, Step=5)
        ↓
  Filter Selection → 47 features (Variance + Correlation + MI)
        ↓
  Genetic Algorithm (DEAP) → 17 features
        ↓
  XGBoost Model → RMSE: 18.553
        ↓
  Feature Store Registration (engine_rul_features)
```

---

## Repository Structure

```
DSAI3202-lab2/
├── lab5_pipeline/
│   ├── components/
│   │   ├── preprocess.py         ← Raw → Gold (RUL + Scaling + Clipping)
│   │   ├── extract_features.py   ← tsfresh rolling window extraction
│   │   ├── select_features.py    ← Filter + GA selection
│   │   └── train_model.py        ← XGBoost training & evaluation
│   └── pipeline.py               ← Azure ML Pipeline submission
├── feature_store/
│   ├── entity.yml                ← EngineEntity definition
│   ├── feature_set.yml           ← Feature Set definition
│   └── spec/
│       └── FeatureSetSpec.yaml   ← Feature specification
└── README.md
```

---

## Dataset

- **Source:** [NASA PCoE Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) — Item 6
- **Subset used:** FD001
- **Train:** 20,631 rows | 100 engines
- **Test:** 13,096 rows | 100 engines
- **Target:** Remaining Useful Life (RUL) in cycles

---

## Azure Data Lake Structure

```
lab5 (container)
├── raw/
│   ├── train_FD001.txt          ← Raw input (never modified)
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── silver/
│   ├── train_FD001.parquet      ← Cleaned + RUL added
│   └── test_FD001.parquet
└── gold/
    ├── train_FD001.parquet           ← Scaled features
    ├── test_FD001.parquet
    ├── features_train_FD001.parquet  ← tsfresh features
    ├── features_filtered_FD001.parquet
    ├── features_final_FD001.parquet  ← GA selected features
    ├── scaler_FD001.pkl
    └── final_report.json
```

---

## Part 1 – Time-Series Exploration

### Data Structure
- 26 columns: engine_id, cycle, 3 operational settings, 21 sensors
- No missing values
- RUL range: 0 – 361 cycles (clipped at 125)

### Sensor Analysis
7 near-constant sensors dropped (variance < 0.01):
`sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19`

14 active sensors retained for feature extraction.

---

## Part 2 – Azure ML Pipeline

### Pipeline Components

The pipeline consists of 4 modular command components built with Azure ML Python SDK v2:

#### Component 1 — Preprocess (`preprocess.py`)
- Reads raw `.txt` files from `lab5/raw/` in Azure Blob Storage
- Assigns column names, casts types, removes constant sensors
- Computes RUL: `max_cycle - current_cycle`
- **RUL Clipping at 125**: clips RUL to focus model on degradation phase only
- Scales features using `StandardScaler` (fit on train only)
- Outputs: `train_gold.parquet`, `test_gold.parquet`

#### Component 2 — Extract Features (`extract_features.py`)
- Builds rolling windows per engine (window=30, **step=5**)
- Extracts time-series features using `tsfresh` `MinimalFCParameters`
- Step size of 5 increases data diversity from 20K to ~4,163 rows with varying RUL labels

| Setting | Value |
|---------|-------|
| Parameters | `MinimalFCParameters` |
| Window size | 30 cycles |
| Step size | 5 cycles |
| Features extracted | 140 |

#### Component 3 — Select Features (`select_features.py`)

| Stage | Method | Features |
|-------|--------|----------|
| Input | — | 140 |
| Stage 1 | Variance Threshold (0.01) | 121 |
| Stage 2 | Correlation Filter (> 0.95) | 47 |
| Stage 3 | Mutual Information (top 50) | 47 |
| Stage 4 | Genetic Algorithm (DEAP) | **17** |

**Genetic Algorithm Config:**

| Parameter | Value |
|-----------|-------|
| Population | 20 |
| Generations | 10 |
| Crossover | Two-point (P=0.6) |
| Mutation | Bit-flip (P=0.2) |
| Fitness | RMSE + 0.01 × (features/total) |
| Model (eval) | DecisionTree depth=5, cv=2 |

#### Component 4 — Train Model (`train_model.py`)
- Trains XGBoost with 300 estimators
- Evaluates on test set
- Saves model as `xgboost_model.json` and metrics as `report.json`

---

## Model Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| **XGBoost (Pipeline)** | **18.553** | **13.117** | **0.547** |
| XGBoost (Notebook) | 46.278 | 35.062 | 0.384 |
| LightGBM | 46.549 | 35.631 | 0.377 |
| RF Tuned | 47.454 | 36.076 | 0.353 |
| Random Forest | 47.605 | 36.242 | 0.348 |

The Pipeline achieved significantly better results due to:
- RUL Clipping at 125 (focused model on degradation phase)
- Step size of 5 (increased data diversity)


<img width="440" height="615" alt="image" src="https://github.com/user-attachments/assets/2154c08a-500c-4b4e-b47e-076c2295556f" />
<img width="1217" height="620" alt="image" src="https://github.com/user-attachments/assets/31d8a9d1-0ade-43a9-bed9-f08bbdb9f312" />

---

## Part 3 – Azure ML Feature Store

### Why Feature Store?
The Feature Store allows features to be versioned, reused across pipelines, and shared across teams without recomputation. Instead of each pipeline recomputing the same features, they are computed once and registered centrally.

### Entity
```yaml
name: EngineEntity
version: "1"
index_columns:
  - name: engine_id
    type: integer
```

### Feature Set
```yaml
name: engine_rul_features
version: "1"
entities:
  - azureml:EngineEntity:1
features:
  - name: RUL
    type: float
```

### Registration Commands
```bash
# Register Entity
az ml feature-store-entity create \
  --file feature_store/entity.yml \
  --resource-group rg-60307052 \
  --feature-store-name amazon-electronics-fs-60307052

# Register Feature Set
az ml feature-set create \
  --file feature_store/feature_set.yml \
  --resource-group rg-60307052 \
  --feature-store-name amazon-electronics-fs-60307052
```

---

## Runtime Summary

| Stage | Time |
|-------|------|
| Data loading & preprocessing | < 1 min |
| tsfresh extraction (train) | ~1.4 min |
| Filter selection | < 0.5 min |
| GA selection | ~0.6 min |
| Model training (XGBoost) | ~1.5 min |
| tsfresh extraction (test) | ~0.8 min |
| **Total** | **~5 minutes** |

---

## How to Reproduce

### Requirements
```bash
pip install azure-ai-ml azure-identity tsfresh deap xgboost scikit-learn pandas azure-storage-blob
```

### Run Pipeline
```bash
py lab5_pipeline/pipeline.py
```

### Config
```python
STORAGE_ACCOUNT = "your_storage_account"
ACCOUNT_KEY     = "your_key"         
CONTAINER       = "lab5"
WINDOW_SIZE     = 30
STEP_SIZE       = 5
RUL_CLIP        = 125
```

---

## Compute Environment

- **Platform:** Azure ML Studio
- **Workspace:** Amazon-Electronics-Lab-60307052
- **Feature Store:** amazon-electronics-fs-60307052
- **Compute:** Serverless (Standard_E8s_v3)
- **Runtime:** Python 3.10

---

## Reflection

### What I Learned

In This lab I was provided a hands-on experience with the full predictive maintenance pipeline from raw sensor data to a registered Feature Store. Key takeaways include:

- **tsfresh** is a powerful library for automated time-series feature engineering, but parameter selection critically impacts runtime. Switching from `EfficientFCParameters` to `MinimalFCParameters` reduced extraction time from over 2 hours to just 1.4 minutes — an 85x speedup.
- **RUL Clipping at 125** was the single most impactful improvement — it reduced RMSE from 46.278 to 18.553 by focusing the model on the degradation phase only.
- **Rolling Window Step Size** matters significantly. Changing from step=1 to step=5 increased data diversity and gave the model more varied RUL labels to learn from.
- **Medallion Architecture** (Raw → Silver → Gold) proved practically valuable for organizing intermediate results and avoiding recomputation.
- **Azure ML Feature Store** enables feature reuse across pipelines and teams, ensuring consistency and reducing redundant computation.
- **Genetic Algorithms** are an effective wrapper method for feature selection, discovering feature combinations that work well together rather than just individually strong features.

---

### Model Results Analysis

The Pipeline XGBoost achieved RMSE of **18.553** and R² of **0.547**, a major improvement over the notebook baseline of 46.278. The two key improvements were RUL clipping at 125 and rolling window step size of 5. The R² of 0.547 means the model explains ~55% of RUL variance — a solid result given only 17 features and no hyperparameter tuning.

---

### Challenges Faced

| Challenge | How It Was Resolved |
|-----------|-------------------|
| `EfficientFCParameters` took 2+ hours | Switched to `MinimalFCParameters` (85x faster) |
| `n_jobs=-1` caused `ValueError` in Spark | Set `n_jobs=1` for Spark compatibility |
| Storage path confusion (`raw` vs `lab5` container) | Restructured to single `lab5` container |
| Feature Store YAML validation errors | Separated `FeatureSetSpec.yaml` into `spec/` subfolder |
| Pipeline sending to Feature Store workspace | Fixed `workspace_name` to ML Workspace |
| RMSE too high (46.278) | Added RUL clipping + step size = 5 |

---

### What Could Be Improved

- Use `EfficientFCParameters` on a more powerful compute cluster to extract richer features
- Tune XGBoost hyperparameters using **Bayesian optimization**
- Extend the pipeline to all four subsets (FD001–FD004) to test generalizability
- Add **MLflow experiment tracking** in Azure ML for reproducible runs
- Implement **feature materialization** in the Feature Store for real-time serving

