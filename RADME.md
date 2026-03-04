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
<img width="1638" height="804" alt="image" src="https://github.com/user-attachments/assets/2c884906-cd13-428f-8178-289d829f93f8" />

### Feature Store Versioning (Schema Evolution)
Registration was performed using the Azure ML CLI to create a governed, searchable asset.

Version 1: Initial feature set including Length, Sentiment, and TF-IDF.

Version 2: Successfully evolved the schema to include the Capital Letters Ratio.



## Reflection
Building this pipeline highlighted that data engineering is 80% of the work in ML. By implementing drift-resistant sampling and a leakage-proof split strategy, the resulting 500+ features are not just numerous, but reliable. Using a Feature Store for Version 2 registration proved how essential versioning is—allowing for feature iteration (adding the Capital Ratio) without breaking the existing Version 1 dependencies.

<img width="1520" height="352" alt="image" src="https://github.com/user-attachments/assets/cb150b37-bac7-4035-bd71-6d748057af1e" />
<img width="1260" height="564" alt="image" src="https://github.com/user-attachments/assets/39c02d0c-3f3a-4351-986a-f576d52ea2f6" />

## VI. Words of Affirmation
"The code worked, the JSON returned, and the cloud obeyed."
