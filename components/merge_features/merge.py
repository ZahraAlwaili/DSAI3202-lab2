import argparse
import os
import pandas as pd
import glob

# Primary keys for identification
KEYS = ["asin", "reviewerID"]

def read_and_clean_df(folder_path: str, name: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet found in {name}")
    
    df = pd.read_parquet(files[0])
    
    # Check for keys
    if not all(k in df.columns for k in KEYS):
        raise KeyError(f"Component {name} is missing keys {KEYS}. Columns: {df.columns.tolist()}")

    # To avoid duplicate column errors, we only keep:
    # 1. The Keys
    # 2. Columns NOT present in the other dataframes (except the keys)
    # For simplicity, if it's not the first dataframe, we strip out common metadata
    if name != "length": # We'll keep metadata from the first one ('length')
        metadata_to_drop = ["price", "brand", "overall", "reviewText", "summary", "verified", "category"]
        cols_to_drop = [c for c in metadata_to_drop if c in df.columns and c not in KEYS]
        if cols_to_drop:
            print(f"Dropping redundant columns from {name}: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--capital_letters", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # Load all
    print("Loading and cleaning datasets...")
    dfs = {
        "length": read_and_clean_df(args.length, "length"),
        "sentiment": read_and_clean_df(args.sentiment, "sentiment"),
        "tfidf": read_and_clean_df(args.tfidf, "tfidf"),
        "sbert": read_and_clean_df(args.sbert, "sbert"),
        "capital": read_and_clean_df(args.capital_letters, "capital_letters")
    }

    # Start merging
    merged = dfs["length"]
    for name in ["sentiment", "tfidf", "sbert", "capital"]:
        print(f"Merging {name} features...")
        # Use suffixes to handle any unexpected overlaps
        merged = merged.merge(dfs[name], on=KEYS, how="inner", suffixes=(None, '_drop'))
        
        # Drop any column that got a suffix due to being a duplicate
        cols_to_drop = [c for c in merged.columns if c.endswith('_drop')]
        if cols_to_drop:
            merged.drop(columns=cols_to_drop, inplace=True)

    # Save final result
    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.parquet")
    merged.to_parquet(output_path, index=False)
    
    print(f"Success! Final merged shape: {merged.shape}")
    print(f"Total columns: {len(merged.columns)}")

if __name__ == "__main__":
    main()