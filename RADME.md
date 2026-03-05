 ### Lab 4: Advanced Text Feature Engineering & Azure ML Feature Store

### Overview
This lab builds a high-level feature engineering pipeline using Azure Machine Learning to transform more than 300,000 Amazon Electronics reviews into numerical features. The pipeline extracts metadata, sentiment, and semantic context, then registers the results as a versioned Feature Set in the Azure ML Feature Store,this lab is critical to learn how to automate feature engineering workflow which makes the whole proccess reproducible and scalable.

### Dataset Exploration & Validation
Before running the pipeline, the dataset was validated in Azure Databricks to ensure the data is suitable for feature extraction.

**Schema Enforcement:** Verified correct data types for `reviewText` (String) and `overall` (Numeric) to prevent errors during vectorization or model fitting .

**Data Quality:** Checked and handled missing values in the composite keys (`asin`, `reviewerID`)as they are used to identify the products and might break the proccess.
**Visual Insights**
- **Rating Distribution:** Strong class imbalance with a heavy skew toward 5-star reviews (~17M).
- **Review Length:** Most reviews are short, but some exceed 10,000 characters, which helps determine padding length for transformer models.

### Drift-Resistant Sampling
**Problem:** Random sampling from a dataset spanning 1996–2018 may overrepresent recent reviews, causing language drift.

**Solution:** Used stratified sampling by year to ensure balanced representation across 1999–2014, improving model robustness to language changes.

### Feature Engineering Components

### 1. Split Dataset
Performed a 70% Train, 15% Validation, 15% Test split before feature extraction.

**Reason:** Prevents data leakage, ensuring validation and test data remain unseen during feature fitting.

### 2. Text Normalization
Used Python `re` regex to:
- Convert text to lowercase
- Remove URLs, HTML tags, numbers, and punctuation

The same preprocessing is applied to all splits to maintain consistent data distribution.

### 3. Metadata & Intensity Features
Extracted structural features:
- `review_length_words`
- `review_length_chars`

**Capital Letters Ratio (V2):**  
Measures the proportion of uppercase characters to capture reviewer intensity (e.g., ALL CAPS emphasis), which is usually lost during lowercasing.

### 4. Sentiment Features (VADER)
Used VADER sentiment analysis to generate:
- `pos`
- `neg`
- `neu`
- `compound`

VADER is optimized for social media and review text, handling emojis, intensifiers, and negations effectively.

### 5. TF-IDF & Semantic Embeddings

**TF-IDF**
- Extracted top 100 n-grams (1,2)
- Captures word importance and local context (e.g., “not good”).

**SBERT Embeddings**
- Used `all-MiniLM-L6-v2`
- Generated 384-dimensional semantic vectors.

**Reason for both:**  
TF-IDF captures keyword importance, while BERT embeddings capture semantic meaning, helping models understand relationships between similar words.

### Pipeline & Feature Store Registration

### Pipeline Execution
All components are connected in a Directed Acyclic Graph (DAG).

Feature extraction steps (Length, Sentiment, TF-IDF, BERT) run in parallel, reducing total execution time.
### Reflection
working on this lab I discoverd how preparing the data can be the moat crucial step in the ML workflow to achive a high quality model, and the pipeline gave me a deeper understanding of the workflow and how to prevent data leakage or data drift over time. 

Finally, using the pipeline and registering features in the Azure ML Feature Store shows  how important reproducibility and versioning are in real-world ML projects.Instead of manual building and managing the features every time, the pipeline ensures that exactly the same steps are consistently applied with every ingestion, which makes experimentation easier and the workflow more reliable.




<img width="1520" height="352" alt="image" src="https://github.com/user-attachments/assets/cb150b37-bac7-4035-bd71-6d748057af1e" />
<img width="1260" height="564" alt="image" src="https://github.com/user-attachments/assets/39c02d0c-3f3a-4351-986a-f576d52ea2f6" />
