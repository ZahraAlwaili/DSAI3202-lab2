# components/extract_features.py
import argparse
import pandas as pd
import numpy as np
import os
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--window_size", type=int, default=30)
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────
    print("📥 Loading Gold data...")
    train_gold = pd.read_parquet(f"{args.input_path}/train_gold.parquet")
    test_gold  = pd.read_parquet(f"{args.input_path}/test_gold.parquet")

    DROP_SENSORS   = ["sensor_1","sensor_5","sensor_6",
                      "sensor_10","sensor_16","sensor_18","sensor_19"]
    ACTIVE_SENSORS = [c for c in train_gold.columns
                      if c.startswith("sensor_")
                      and c not in DROP_SENSORS]

    # ── Build Windows ────────────────────────────────────
    def build_windows(df):
        rows = []
        for eng_id, grp in df.groupby("engine_id"):
            grp = grp.sort_values("cycle").reset_index(drop=True)
            for end_idx in range(len(grp)):
                start_idx = max(0, end_idx - args.window_size + 1)
                window = grp.iloc[start_idx:end_idx+1].copy()
                window["id"]   = eng_id * 10_000 + end_idx
                window["time"] = range(len(window))
                rows.append(window)
        ts_df = pd.concat(rows, ignore_index=True)
        return ts_df[["id", "time"] + ACTIVE_SENSORS]

    print("🔄 Building train windows...")
    ts_train = build_windows(train_gold)
    print(f"   Shape: {ts_train.shape}")

    print("🔄 Building test windows...")
    ts_test = build_windows(test_gold)
    print(f"   Shape: {ts_test.shape}")

    # ── Extract Features ─────────────────────────────────
    print("⚙️  Extracting train features...")
    X_train = extract_features(
        ts_train,
        column_id="id", column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=1, impute_function=impute,
        disable_progressbar=False
    )

    print("⚙️  Extracting test features...")
    X_test = extract_features(
        ts_test,
        column_id="id", column_sort="time",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=1, impute_function=impute,
        disable_progressbar=False
    )

    # ── Add RUL ──────────────────────────────────────────
    y_train = (train_gold
               .assign(id=lambda d: d["engine_id"] * 10_000 +
                       d.groupby("engine_id").cumcount())
               .set_index("id")["RUL"])
    X_train["RUL"] = y_train.reindex(X_train.index)

    y_test = (test_gold
              .assign(id=lambda d: d["engine_id"] * 10_000 +
                      d.groupby("engine_id").cumcount())
              .set_index("id")["RUL"])
    X_test["RUL"] = y_test.reindex(X_test.index)

    # ── Save ─────────────────────────────────────────────
    os.makedirs(args.output_path, exist_ok=True)
    X_train.reset_index().to_parquet(f"{args.output_path}/features_train.parquet", index=False)
    X_test.reset_index().to_parquet(f"{args.output_path}/features_test.parquet",   index=False)

    print(f"✅ Train features: {X_train.shape}")
    print(f"✅ Test features : {X_test.shape}")
    print("✅ Feature extraction complete!")

if __name__ == "__main__":
    main()