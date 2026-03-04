## Lab 4: Advanced Text Feature Engineering & Azure ML Feature Store
## Overview
This lab implements a production-grade feature engineering pipeline using Azure Machine Learning to transform 300,000+ raw Amazon Electronics reviews into high-signal numerical features. The pipeline automates the extraction of structural metadata, emotional polarity, and semantic context, culminating in a versioned Feature Set registered within the Azure ML Feature Store.

## Dataset Exploration & Validation
Before pipeline execution, data integrity was verified using Azure Databricks to ensure the "Gold" layer was suitable for high-dimensional feature extraction.

Schema Enforcement: Confirmed reviewText (String) and overall (Numeric) types to prevent pipeline crashes during vectorization.

Data Quality: Identified and handled missing values in the composite keys (asin, reviewerID).

Visualizations:

Rating Distribution: Observed a heavy skew toward 5-star ratings (~17M reviews), identifying a class imbalance that will require weighting in future modeling.

Review Length Distribution: Analyzed character counts to determine optimal padding for transformer models, noting a "long tail" of reviews exceeding 10,000 characters.

## Drift-Resistant Sampling
Problem: Random sampling from a 20-year dataset (1996-2018) risks over-representing recent years where review volume is higher, leading to "Language Drift" where models fail on older linguistic patterns.
Solution: Implemented Stratified Sampling by year. By ensuring equal representation across the 1999–2014 timespan, the features remain robust against evolving slang and product categories.

## Feature Engineering Components
Each component is a modular Python script wrapped in a component.yml definition, running on isolated Azure ML compute clusters.

### 1. Split Dataset Component
Purpose: The most critical step for model integrity—preventing data leakage.

Logic: Implemented a two-stage split (70% Train, 15% Val, 15% Test).

Why? Splitting before any feature fitting (like TF-IDF or Scaling) ensures that the validation and test sets remain "unseen" by the feature extractors.

### 2. Normalize Text Component
Logic: Applied standard regex patterns via the re library to lowercase text, remove noise (URLs, HTML tags, numbers), and strip punctuation.

Consistency: The same normalization logic is applied in parallel to all three splits to ensure the data distribution remains identical during inference.

### 3. Metadata & Intensity Features
Review Length: Created review_length_words and review_length_chars to capture reviewer engagement levels.

Capital Letters Ratio (V2 Update): Added in Version 2, this feature calculates the ratio of uppercase characters.

Significance: This captures "Reviewer Intensity" (e.g., shouting in all caps), a nuanced signal often lost when text is lowercased during standard NLP normalization.

### 4. Sentiment Features (VADER)
Logic: Utilized the VADER (Valence Aware Dictionary and sEntiment Reasoner).

Output: Generated pos, neg, neu, and compound scores.

Why VADER? Unlike basic polarity, VADER is specifically tuned for social media and product reviews, handling emojis, intensifiers ("very good!"), and negations ("not bad").

### 5. TF-IDF & Semantic Embeddings
TF-IDF: Extracted top 100 n-grams (1,2). This provides statistical word importance while bigrams capture local context (e.g., "not great").

SBERT Embeddings: Used the all-MiniLM-L6-v2 transformer model to generate 384-dimensional dense vectors.

Why both? TF-IDF captures specific keyword importance (lexical), while BERT captures the "meaning" behind the words (semantic), allowing the model to understand that "excellent" and "superb" are related.

## Pipeline & Feature Store Registration
### Pipeline Execution
The pipeline wires these components into a Directed Acyclic Graph (DAG).

Command: az ml job create --file pipelines/feature_pipeline.yml

Optimization: Feature extraction components (Length, Sentiment, TF-IDF, BERT) run in parallel to minimize total execution time.

### Feature Store Versioning (Schema Evolution)
Registration was performed using the Azure ML CLI to create a governed, searchable asset.

Version 1: Initial feature set including Length, Sentiment, and TF-IDF.

Version 2: Successfully evolved the schema to include the Capital Letters Ratio.

Verification:

PowerShell
az ml feature-set show --name amazon_review_text_features --version 2 --resource-group rg-60307052 --feature-store-name amazon-electronics-fs-60307052
## Reflection
Building this pipeline highlighted that data engineering is 80% of the work in ML. By implementing drift-resistant sampling and a leakage-proof split strategy, the resulting 500+ features are not just numerous, but reliable. Using a Feature Store for Version 2 registration proved how essential versioning is—allowing for feature iteration (adding the Capital Ratio) without breaking the existing Version 1 dependencies.

<img width="1919" height="1065" alt="image" src="https://github.com/user-attachments/assets/c4c6b376-f781-4105-8356-61f00cd0392e" />

<img width="1918" height="1063" alt="image" src="https://github.com/user-attachments/assets/899dbb78-f1d4-467b-b724-b8157b6e368f" />

<img width="1918" height="1079" alt="image" src="https://github.com/user-attachments/assets/9653a038-3cab-492b-be31-ed3d671709c5" />

<img width="1823" height="1014" alt="image" src="https://github.com/user-attachments/assets/7dfe2075-0716-45d1-8673-08d288aba07a" />

<img width="1918" height="997" alt="zahra 1  Screenshot 2026-01-25 101927" src="https://github.com/user-attachments/assets/83f9213e-69a0-4c25-9b3a-38b6e8e6f602" />

<img width="1875" height="1026" alt="zahra 2 Screenshot 2026-01-25 101959" src="https://github.com/user-attachments/assets/452fd24f-de16-4a3c-828e-a11ecb8845f3" />

