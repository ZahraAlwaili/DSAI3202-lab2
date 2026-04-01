import os
import json
import joblib
import pandas as pd
import numpy as np
import logging

model = None

def init():
    """Initialize the model when the deployment starts."""
    global model
    # The model.pkl is always located in 'AZUREML_MODEL_DIR'
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
    )
    
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")

def build_features_for_inference(df):
    """Mirror the Run 2 logic: BERT Embeddings + Numeric TF-IDF/Metadata."""
    to_drop = ['asin', 'reviewerID', 'overall', 'label']
    all_cols = df.columns.tolist()
    
    # 1. Identify BERT features (using 'bert_embedding' found in your logs)
    bert_feats = [c for c in all_cols if 'bert_embedding' in c.lower()]
    
    # 2. Identify all other numeric features (TF-IDF words + Metadata)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    other_feats = [c for c in numeric_cols if c not in to_drop]
    
    # Combine and sort to ensure feature order matches training exactly
    final_features = sorted(list(set(bert_feats + other_feats)))
    
    X = df[final_features]
    return X.fillna(0)

def run(raw_data):
    """Handle prediction requests."""
    try:
        # Parse input JSON
        content = json.loads(raw_data)
        input_data = content["data"]
        
        # Convert to DataFrame and process features
        df = pd.DataFrame(input_data)
        X = build_features_for_inference(df)
        
        # Generate Predictions
        preds = model.predict(X)
        
        return {"predictions": preds.tolist()}
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return {"error": str(e)}