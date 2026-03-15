# components/preprocess.py
import argparse
import pandas as pd
from azure.storage.blob import BlobServiceClient
import io

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage_account", type=str)
    parser.add_argument("--account_key",     type=str)
    parser.add_argument("--container",       type=str, default="lab5")
    parser.add_argument("--output_path",     type=str)
    args = parser.parse_args()

    # ── Connect ──────────────────────────────────────────
    blob_service = BlobServiceClient(
        account_url=f"https://{args.storage_account}.blob.core.windows.net",
        credential=args.account_key
    )

    def read_raw(filename):
        data = blob_service.get_blob_client(
            container=args.container,
            blob=f"raw/{filename}"
        ).download_blob().readall()
        return io.BytesIO(data)

    # ── Load ─────────────────────────────────────────────
    COLUMN_NAMES = (
        ["engine_id", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"sensor_{i}"     for i in range(1, 22)]
    )

    print("📥 Loading raw data...")
    train_df = pd.read_csv(read_raw("train_FD001.txt"),
                           sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df  = pd.read_csv(read_raw("test_FD001.txt"),
                           sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_df   = pd.read_csv(read_raw("RUL_FD001.txt"),
                           sep=r"\s+", header=None, names=["RUL"])

    # ── Add RUL ──────────────────────────────────────────
    max_cycle = (train_df.groupby("engine_id")["cycle"]
                 .max().reset_index()
                 .rename(columns={"cycle": "max_cycle"}))
    train_df = train_df.merge(max_cycle, on="engine_id")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]
    train_df.drop(columns=["max_cycle"], inplace=True)

    rul_df["engine_id"] = range(1, len(rul_df) + 1)
    last_cycle = (test_df.groupby("engine_id")["cycle"]
                  .max().reset_index()
                  .rename(columns={"cycle": "max_cycle"}))
    last_cycle = last_cycle.merge(rul_df, on="engine_id")
    test_df = test_df.merge(last_cycle, on="engine_id")
    test_df["RUL"] = test_df["RUL"] + (test_df["max_cycle"] - test_df["cycle"])
    test_df.drop(columns=["max_cycle"], inplace=True)

    # ── Scale ────────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler

    DROP_SENSORS = ["sensor_1","sensor_5","sensor_6",
                    "sensor_10","sensor_16","sensor_18","sensor_19"]
    FEATURE_COLS = [c for c in train_df.columns
                    if (c.startswith("sensor_") or c.startswith("op_setting_"))
                    and c not in DROP_SENSORS]

    scaler = StandardScaler()
    train_df[FEATURE_COLS] = scaler.fit_transform(train_df[FEATURE_COLS])
    test_df[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    # ── Save ─────────────────────────────────────────────
    import os
    os.makedirs(args.output_path, exist_ok=True)
    train_df.to_parquet(f"{args.output_path}/train_gold.parquet", index=False)
    test_df.to_parquet(f"{args.output_path}/test_gold.parquet",   index=False)

    print(f"✅ Train: {train_df.shape}")
    print(f"✅ Test : {test_df.shape}")
    print("✅ Preprocess complete!")

if __name__ == "__main__":
    main()