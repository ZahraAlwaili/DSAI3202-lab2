# Azure Data Ingestion Pipeline – Amazon Electronics Reviews

## Overview
This project implements a data ingestion pipeline on Azure using the Amazon Electronics Reviews dataset. Raw JSON data is stored in Azure Data Lake Storage (ADLS Gen2), transformed using Azure Data Factory (ADF), and written to Parquet format partitioned by review year.

The lab demonstrates both portal-based and terminal-based ingestion workflows used in real-world data engineering.

---

## Technologies Used
- Azure Blob Storage (ADLS Gen2)
- Azure Data Factory (ADF)
- Azure ML Compute Instance (VM)
- Azure CLI & AzCopy
- Python

---

## Dataset
Amazon Electronics dataset from Stanford SNAP:
- reviews_Electronics_5.json
- meta_Electronics.json

---

## Data Lake Structure
- raw/ – Original datasets
- processed/ – Transformed Parquet data
- curated/ – Reserved for analytics

---

## Data Ingestion

### Product Metadata
- Uploaded to the `raw` container using the Azure Portal
- Converted to valid line-delimited JSON and re-uploaded as:


<img width="1919" height="1065" alt="image" src="https://github.com/user-attachments/assets/c4c6b376-f781-4105-8356-61f00cd0392e" />

<img width="1918" height="1063" alt="image" src="https://github.com/user-attachments/assets/899dbb78-f1d4-467b-b724-b8157b6e368f" />


<img width="1918" height="1079" alt="image" src="https://github.com/user-attachments/assets/9653a038-3cab-492b-be31-ed3d671709c5" />


<img width="1823" height="1014" alt="image" src="https://github.com/user-attachments/assets/7dfe2075-0716-45d1-8673-08d288aba07a" />

<img width="1918" height="997" alt="zahra 1  Screenshot 2026-01-25 101927" src="https://github.com/user-attachments/assets/83f9213e-69a0-4c25-9b3a-38b6e8e6f602" />




<img width="1875" height="1026" alt="zahra 2 Screenshot 2026-01-25 101959" src="https://github.com/user-attachments/assets/452fd24f-de16-4a3c-828e-a11ecb8845f3" />

