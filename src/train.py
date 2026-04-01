import argparse
import os
import time
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

def load_data(path):
    # Path logic for Azure ML folder inputs
    parquet_path = os.path.join(path, "data.parquet")
    return pd.read_parquet(parquet_path)

def create_labels(df):
    # 4-5 stars = Positive (1), 1-3 stars = Negative (0)
    df["label"] = (df["overall"] >= 4).astype(int)
    return df
def build_features(df, run_type):
    # 1. Targets and IDs to drop
    to_drop = ['asin', 'reviewerID', 'overall', 'label']
    
    all_cols = df.columns.tolist()
    
    if run_type == "run1":
        # Run 1: BERT Embeddings Only (Found 'bert_embedding' in your logs)
        features = [c for c in all_cols if 'bert_embedding' in c.lower()]
        X = df[features]
    elif run_type == "run2":
        # Run 2: BERT + TF-IDF 
        # Note: In your data, TF-IDF features are usually the word columns 
        # that aren't metadata. We'll include bert + all other numeric features.
        bert_feats = [c for c in all_cols if 'bert_embedding' in c.lower()]
        # Getting numeric columns that aren't BERT or the targets
        other_feats = df.select_dtypes(include=[np.number]).columns.tolist()
        features = list(set(bert_feats + [c for c in other_feats if c not in to_drop]))
        X = df[features]
    else:
        # Run 3: All numeric features (Metadata + Embeddings + Words)
        X = df.drop(columns=[c for c in to_drop if c in df.columns])
        X = X.select_dtypes(include=[np.number])

    # Safety check
    if X.shape[1] == 0:
        raise ValueError(f"Zero features found for {run_type}. Check column names!")

    print(f"[{run_type.upper()}] Success! Loaded {X.shape[1]} features.")
    return X.fillna(0)

def evaluate_and_log(model, X, y, split):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    metrics = {
        f"{split}_accuracy": accuracy_score(y, preds),
        f"{split}_auc": roc_auc_score(y, probs),
    }
    for name, value in metrics.items():
        mlflow.log_metric(name, value)
        print(f"{name}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--run_type", type=str, default='run1')
    parser.add_argument("--C", type=float, default=10.0)
    parser.add_argument("--solver", type=str, default='liblinear')
    args = parser.parse_args()

    mlflow.start_run()
    
    # Load and Process
    train_df = create_labels(load_data(args.train_data))
    val_df = create_labels(load_data(args.val_data))
    test_df = create_labels(load_data(args.test_data))

    X_train, y_train = build_features(train_df, args.run_type), train_df["label"]
    X_val, y_val = build_features(val_df, args.run_type), val_df["label"]
    X_test, y_test = build_features(test_df, args.run_type), test_df["label"]

    # Train
    model = LogisticRegression(C=args.C, solver=args.solver, max_iter=1000)
    model.fit(X_train, y_train)

    # Log
    evaluate_and_log(model, X_train, y_train, "train")
    evaluate_and_log(model, X_val, y_val, "val")
    evaluate_and_log(model, X_test, y_test, "test")

    # Save
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(model, os.path.join(args.output, "model.pkl"))
    mlflow.end_run()

if __name__ == "__main__":
    main()