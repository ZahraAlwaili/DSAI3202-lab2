import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deployment_out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset from parquet
    df = pd.read_parquet(args.data)

    # 1. Sort by year to ensure Deployment is the "future" data (Data Drift simulation)
    if 'review_year' in df.columns:
        df = df.sort_values("review_year", ascending=True)
    else:
        print("Warning: review_year not found. Falling back to random split for deployment.")

    # 2. Slice off the last 10% for Deployment
    deploy_size = int(len(df) * 0.10)
    deployment_df = df.iloc[-deploy_size:]
    remaining_df = df.iloc[:-deploy_size]

    # 3. Split the remaining 90% into:
    # Train (60% of total), Val (15% of total), Test (15% of total)
    # Math: test_size=0.3333 on the remaining 90% leaves 60% of original for train
    train_df, temp_df = train_test_split(
        remaining_df,
        test_size=0.3333, 
        random_state=args.seed,
        shuffle=True
    )

    # 4. Split the temp (30% of total) into Val (15%) and Test (15%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5, 
        random_state=args.seed,
        shuffle=True
    )

    # Create directories for outputs
    for path in [args.train_out, args.val_out, args.test_out, args.deployment_out]:
        os.makedirs(path, exist_ok=True)

    # Write outputs as parquet
    train_df.to_parquet(os.path.join(args.train_out, "data.parquet"))
    val_df.to_parquet(os.path.join(args.val_out, "data.parquet"))
    test_df.to_parquet(os.path.join(args.test_out, "data.parquet"))
    deployment_df.to_parquet(os.path.join(args.deployment_out, "data.parquet"))

    print(f"Success! Total rows: {len(df)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)} | Deploy: {len(deployment_df)}")

if __name__ == "__main__":
    main()