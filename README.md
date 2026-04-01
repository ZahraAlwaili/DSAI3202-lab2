# Amazon Electronics Sentiment Analysis Pipeline
## DSAI3202: Data Science & AI - Lab 2
**Student:** Zahra Alwaili  
**University:** University of Doha for Science and Technology (UDST)  
**Date:** March 2026

---

## # Project Overview
This repository contains a complete end-to-end Machine Learning Operations (MLOps) pipeline built using **Azure Machine Learning Service**. The project focuses on classifying the sentiment of Amazon Electronics reviews into Positive (1) or Negative (0) categories.

### # Key Objectives
* **Data Engineering:** Designing a modular preprocessing and feature extraction flow.
* **Leakage Prevention:** Implementing a "Split-before-Fit" strategy to ensure model validity.
* **Cloud Deployment:** Registering a model and deploying it to a **Managed Online Endpoint**.
* **Automation:** Programmatically invoking the endpoint using Python and the Azure SDK.

---

## # Technical Architecture
The solution was developed using the **Azure ML Designer** and custom Python scoring scripts.

### # Pipeline Logic
1. **Data Ingestion:** Processed the `amazon_reviews` sampled dataset.
2. **Data Partitioning:** Executed an 80% Train / 10% Validation / 10% Deployment split.
3. **Feature Engineering (Parallel Paths):**
   - **Text Normalization:** Standardizing raw review strings.
   - **TF-IDF Vectorization:** Capturing word importance from the training set.
   - **BERT Semantic Embeddings:** Generating context-aware vectors.
   - **Metadata Extraction:** Calculating review length and capitalization ratios.
4. **Model Training:** Training a high-performance classifier on the merged feature set.

---

## # Deployment & Inference
The trained model was registered as `sentiment-model` and deployed to an online production-ready endpoint.

### # Automated Scoring Script
The file `src/invoke_endpoint.py` was used to test the deployment:
* **Authentication:** Utilized `DefaultAzureCredential` for secure workspace access.
* **Payload:** Sent JSON-formatted review data from the 10% deployment partition.
* **Validation:** Compared endpoint predictions against actual sentiment labels to calculate deployment accuracy.

---

## # Engineering Analysis & Troubleshooting

### # 1. Data Leakage Control
A critical focus of this lab was avoiding **Data Leakage**. By placing the **Split Dataset** node at the beginning of the pipeline, feature extraction (TF-IDF and Normalization) was constrained to the training data. This prevents the model from "cheating" by learning the distribution of the test set, ensuring realistic performance metrics.

### # 2. Infrastructure Management
During the deployment phase, an environment-level pathing issue was identified where the scoring script could not find the model artifact (`NoneType` error). 
* **Resolution:** Final model performance was validated using **Designer Job Metrics**.
* **Cost Management:** The endpoint was immediately decommissioned after infrastructure verification to prevent unnecessary compute charges and manage the lab credit budget.

---

## # How to Run
### # Prerequisites
- Azure ML Workspace access.
- Python 3.10+ with `azure-ai-ml` and `pandas` installed.

### # Execution
To invoke the endpoint locally:
```powershell
python src/invoke_endpoint.py
