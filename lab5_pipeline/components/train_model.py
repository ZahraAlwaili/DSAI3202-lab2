# components/train_model.py
import argparse
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────
    print("📥 Loading final features...")
    train = pd.read_parquet(f"{args.input_path}/train_final.parquet")
    test  = pd.read_parquet(f"{args.input_path}/test_final.parquet")

    y_train = train["RUL"]
    X_train = train.drop(columns=["RUL"])

    y_test  = test["RUL"].dropna()
    X_test  = test.drop(columns=["RUL"]).loc[y_test.index]

    print(f"✅ Train: {X_train.shape}")
    print(f"✅ Test : {X_test.shape}")

    # ── Train ─────────────────────────────────────────────
    print("\n🤖 Training XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    # ── Evaluate ──────────────────────────────────────────
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)

    print("\n" + "=" * 45)
    print("📊 FINAL RESULTS — Lab 5 Pipeline")
    print("=" * 45)
    print(f"   RMSE : {rmse:.3f}")
    print(f"   MAE  : {mae:.3f}")
    print(f"   R²   : {r2:.3f}")
    print("=" * 45)

    # ── Save ──────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)

    model.save_model(f"{args.output_path}/xgboost_model.json")

    report = {
        "timestamp" : datetime.now().isoformat(),
        "model"     : "XGBoost",
        "rmse"      : round(rmse, 3),
        "mae"       : round(mae, 3),
        "r2"        : round(r2, 3),
        "features"  : X_train.shape[1],
    }
    with open(f"{args.output_path}/report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Model saved!")
    print("✅ Training complete!")

if __name__ == "__main__":
    main()