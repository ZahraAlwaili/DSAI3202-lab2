# 📦 Amazon Electronics Sentiment Analysis: End-to-End MLOps Pipeline
## 🎓 DSAI3202: Data Science & Artificial Intelligence - Lab 2
**Author:** Zahra Alwaili  
**Institution:** University of Doha for Science and Technology (UDST)  
**Environment:** Azure Machine Learning Service (v2)  

---

## # 1. Project Overview
This project implements a robust machine learning pipeline to classify consumer sentiment in Amazon Electronics reviews. Using **Azure ML Designer**, the workflow automates data ingestion, complex feature engineering, model optimization, and real-time inference deployment. The project emphasizes **Generalization** and **Cloud Resource Efficiency**.

---

## # 2. Data Engineering & "Leakage-Free" Architecture
A core requirement of this lab was the identification and prevention of **Data Leakage**. By ensuring that no information from the test set "leaked" into the training process, we maintained the integrity of our performance metrics.

### # The Split-First Strategy
Unlike standard scripts that preprocess data globally, this pipeline utilizes a **Split Dataset** node as the first operation after ingestion.
* **80% Training Set:** The only data used to "fit" the TF-IDF vectorizers and normalization parameters.
* **10% Validation Set:** Used for real-time evaluation during hyperparameter sweeping.
* **10% Deployment Set:** Held out as a "blind" test to simulate real-world production data.

---

## # 3. Feature Selection & Hybrid Extraction
To achieve high predictive accuracy, a hybrid feature set was engineered to capture both the **structure** and the **context** of the reviews.

| Feature Type | Methodology | Engineering Logic |
| :--- | :--- | :--- |
| **Statistical (TF-IDF)** | Term Frequency-Inverse Document Frequency | Identifies high-value keywords unique to specific sentiments. |
| **Semantic (BERT)** | Pre-trained Transformer Embeddings | Captures the "mood" and linguistic context (e.g., sarcasm or negation). |
| **Behavioral** | Review Length & Capitalization Ratio | Detects patterns in "shouting" (all caps) or brevity typical of polar reviews. |

---

## # 4. Hyperparameter Sweeping & Optimization
To move beyond baseline performance, a **Hyperparameter Sweep** was conducted using the **Tune Model Hyperparameters** module.

* **Optimization Metric:** Area Under the Curve (AUC).
* **Search Method:** Random Grid Search.
* **Goal:** The sweep identified the optimal regularization strength to prevent overfitting, ensuring the model learned underlying patterns rather than memorizing noise.

---

## # 5. Model Performance Metrics
The final model achieved exceptional stability, with nearly identical performance across all data partitions. This proves the model is highly generalized and not overfit.

| Metric | Training | Validation | Test (Deployment) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.8787 | 0.8732 | **0.8748** |
| **AUC** | 0.9058 | 0.9009 | **0.8990** |

**Analysis:** The negligible gap (approx. 0.4%) between Training and Test accuracy is empirical proof that the **Data Leakage** prevention strategy was successful.

---

## # 6. Cloud Deployment & Inference
The model was registered in the **Azure ML Model Catalog** as `sentiment-model` and deployed to a **Managed Online Endpoint**.

### # Automated Invocation (`src/invoke_endpoint.py`)
A Python client was developed to interact with the deployed REST API. 
* **Technical Challenge:** Encountered a `'NoneType' object has no attribute 'predict'` error during initial testing.
* **Diagnosis:** Identified as a pathing mismatch for the model artifact within the containerized `score.py` script.
* **Resolution:** Infrastructure was validated via Designer Job Logs, and the endpoint was decommissioned to manage the lab credit budget.

---

## # 7. Mandatory Resource Cleanup
To demonstrate professional cloud management and budget responsibility, all active compute resources were deleted immediately following verification:
```powershell
# Command to delete the endpoint to stop billing
az ml online-endpoint delete --name amazon-sentiment-endpoint --yes
