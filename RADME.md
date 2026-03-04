## Lab 4: Text Feature Engineering & Azure ML Feature Store
## I. Introduction
This project implements a production-grade feature engineering pipeline using Azure Machine Learning. We have transformed the raw Amazon Electronics reviews dataset into a versioned, multi-dimensional feature set. By moving from raw text to numerical representations, we have prepared the data for high-performance machine learning models.

## II. Exploration, Validation, and Sampling
Before building the pipeline, we conducted an initial validation in Databricks:

Schema Verification: Confirmed reviewText is stored as strings and overall ratings are numeric.

Sampling for Scale: Created a sampled subset of 300,000 reviews to balance feature richness with computational efficiency.

Drift Resistance: The sampling strategy ensures a diverse representation of reviews to remain resistant to language evolution and temporal drift.

## III. Azure ML Feature Engineering Pipeline
The pipeline is built using Azure ML Command Components, ensuring a modular and reproducible workflow. Each step is tracked, and components run on dedicated Azure ML compute clusters.

### Feature Engineering Components
Split Dataset: Performs a 70/15/15 (Train/Val/Test) split to prevent data leakage.

Normalize Text: Cleans reviews by removing URLs, punctuation, and converting to lowercase for specific NLP tasks.

Review Length: Calculates word and character counts to capture the level of detail provided by the reviewer.

Sentiment Analysis: Uses VADER to extract Pos/Neg/Neu and Compound polarity scores.

TF-IDF: Captures term importance using N-grams (1,2) to represent word frequency and relevance.

SBERT Embeddings: Generates dense semantic vectors using transformer-based models to capture deep contextual meaning.

Capital Letters Ratio: Measures the proportion of uppercase characters. This serves as a proxy for "intensity" or "shouting," providing signal that standard lowercase normalization might miss.

## IV. Feature Store Registration & Versioning
The final output is registered in the Azure ML Feature Store, decoupling data engineering from model training.

### Entity Definition
Name: AmazonReview

Index Columns: reviewerID, asin (Composite Key)

Version: 2

### Schema Evolution (Versioning)
We utilized a versioning strategy to iterate on our feature set:

Version 1: Initial feature set including length, sentiment, TF-IDF, and SBERT.

Version 2: (Current) Integrated the Capital Letters Ratio feature and updated the Feature Set Specification to reflect the expanded schema.

## V. CLI Verification
Success was verified using the Azure CLI. The following command confirms that Version 2 is live with all features registered:

PowerShell
az ml feature-set show --name amazon_review_text_features --version 2 --resource-group rg-60307052 --feature-store-name amazon-electronics-fs-60307052
## VI. Words of Affirmation
"The code worked, the JSON returned, and the cloud obeyed."
