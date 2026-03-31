import argparse
import os
import time
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score
)

def load_data(path):
    parquet_path = os.path.join(path, "data.parquet")
    return pd.read_parquet(parquet_path)

def create_labels(df):
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df, run_type):
    # 1. Start with targets and IDs to drop
    to_drop = ['asin', 'reviewerID', 'overall', 'label']
    
    if run_type == "run1":
        # SBERT Only: Keep columns starting with 'sbert'
        features = [c for c in df.columns if c.startswith('sbert')]
        X = df[features]
    elif run_type == "run2":
        # SBERT + TF-IDF: Keep columns starting with 'sbert' or 'tfidf'
        features = [c for c in df.columns if c.startswith('sbert') or c.startswith('tfidf')]
        X = df[features]
    else:
        # Run 3 / Default: All numeric features
        X = df.drop(columns=[c for c in to_drop if c in df.columns])
        X = X.select_dtypes(include=[np.number])
    
    # Handle NaNs
    X = X.fillna(0)
    
    print(f"[{run_type.upper()}] Feature matrix shape: {X.shape}")
    return X

def evaluate_and_log(model, X, y, split):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    metrics = {
        f"{split}_accuracy": accuracy_score(y, preds),
        f"{split}_auc": roc_auc_score(y, probs),
        f"{split}_precision": precision_score(y, preds),
        f"{split}_recall": recall_score(y, preds),
        f"{split}_f1": f1_score(y, preds)
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"{name}: {value:.4f}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # New Argument for Feature Experiments
    parser.add_argument("--run_type", type=str, default='run3', help="run1, run2, or run3")
    
    # Hyperparameters
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--solver", type=str, default='liblinear')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    mlflow.start_run()
    mlflow.log_param("C", args.C)
    mlflow.log_param("solver", args.solver)
    mlflow.log_param("feature_run_type", args.run_type)
    
    start_time = time.time()

    print(f"Loading datasets for {args.run_type}...")
    train_df = create_labels(load_data(args.train_data))
    val_df = create_labels(load_data(args.val_data))
    test_df = create_labels(load_data(args.test_data))

    X_train = build_features(train_df, args.run_type)
    y_train = train_df["label"]
    
    X_val = build_features(val_df, args.run_type)
    y_val = val_df["label"]
    
    X_test = build_features(test_df, args.run_type)
    y_test = test_df["label"]

    print(f"Training Model ({args.run_type})...")
    model = LogisticRegression(C=args.C, solver=args.solver, max_iter=1000)
    model.fit(X_train, y_train)

    evaluate_and_log(model, X_train, y_train, "train")
    evaluate_and_log(model, X_val, y_val, "val")
    evaluate_and_log(model, X_test, y_test, "test")

    total_runtime = time.time() - start_time
    mlflow.log_metric("total_training_runtime_seconds", total_runtime)

    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    
    mlflow.end_run()

if __name__ == "__main__":
    main()