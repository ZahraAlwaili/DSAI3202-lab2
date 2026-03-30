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
    # Divine Intervention: Points to the folder, reads the parquet inside
    parquet_path = os.path.join(path, "data.parquet")
    return pd.read_parquet(parquet_path)

def create_labels(df):
    # Binary classification: 4-5 stars = 1 (Positive), 1-3 stars = 0 (Negative)
    df["label"] = (df["overall"] >= 4).astype(int)
    return df

def build_features(df):
    # 1. Drop targets and IDs
    to_drop = ['asin', 'reviewerID', 'overall', 'label']
    X = df.drop(columns=[c for c in to_drop if c in df.columns])
    
    # 2. Keep ONLY numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # 3. Handle NaNs
    X = X.fillna(0)
    
    print(f"Feature matrix shape: {X.shape} | NaNs remaining: {X.isna().sum().sum()}")
    return X

def evaluate_and_log(model, X, y, split):
    """
    Logs all metrics for the specified split.
    Note: 'split' will be 'train', 'val', or 'test'.
    """
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # These names (e.g., val_accuracy) must match your sweep_job.yml primary_metric
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
    # Data paths
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    # Hyperparameters for Tuning
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default='liblinear')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Start MLflow Run
    mlflow.start_run()
    
    # Log hyperparameters so Azure ML can track them
    mlflow.log_param("C", args.C)
    mlflow.log_param("solver", args.solver)
    
    start_time = time.time()

    print("Loading datasets...")
    train_df = create_labels(load_data(args.train_data))
    val_df = create_labels(load_data(args.val_data))
    test_df = create_labels(load_data(args.test_data))

    X_train = build_features(train_df)
    y_train = train_df["label"]
    
    X_val = build_features(val_df)
    y_val = val_df["label"]
    
    X_test = build_features(test_df)
    y_test = test_df["label"]

    print(f"Training Model with C={args.C}, solver={args.solver}...")
    
    # Use args.C and args.solver passed from the Sweep Job
    model = LogisticRegression(
        C=args.C, 
        solver=args.solver, 
        max_iter=1000
    )
    model.fit(X_train, y_train)

    print("Logging all metrics...")
    evaluate_and_log(model, X_train, y_train, "train")
    evaluate_and_log(model, X_val, y_val, "val")  # This logs 'val_accuracy'
    evaluate_and_log(model, X_test, y_test, "test")

    # Log runtime
    total_runtime = time.time() - start_time
    mlflow.log_metric("total_training_runtime_seconds", total_runtime)
    print(f"Total Runtime: {total_runtime:.2f}s")

    # Save artifact
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    
    mlflow.end_run()

if __name__ == "__main__":
    main()