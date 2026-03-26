import argparse
import os
import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split

# Primary keys for identification
KEYS = ["asin", "reviewerID"]

def flatten_vectors(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Expands a column of lists/arrays into separate numeric columns."""
    if col_name in df.columns:
        # Check if the first entry is a list or numpy array
        first_val = df[col_name].iloc[0]
        if isinstance(first_val, (list, np.ndarray)):
            print(f"Flattening vector column: {col_name}")
            # Expand the list into a new DataFrame
            expanded_df = pd.DataFrame(df[col_name].tolist(), index=df.index)
            # Add prefix to new columns (e.g., sbert_0, sbert_1)
            expanded_df.columns = [f"{col_name}_{i}" for i in range(expanded_df.shape[1])]
            # Drop the original 'list' column and join the flattened ones
            df = pd.concat([df.drop(columns=[col_name]), expanded_df], axis=1)
    return df

def read_and_clean_df(folder_path: str, name: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet found in {name}")
    
    df = pd.read_parquet(files[0])
    
    if not all(k in df.columns for k in KEYS):
        raise KeyError(f"Component {name} is missing keys {KEYS}")

    # Metadata columns to remove (keeping 'overall' as the label)
    if name != "length": 
        metadata_to_drop = ["price", "brand", "reviewText", "summary", "verified", "category", "review_year"]
        cols_to_drop = [c for c in metadata_to_drop if c in df.columns and c not in KEYS]
        if cols_to_drop:
            print(f"Dropping redundant columns from {name}: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
    
    # --- FIX: Flatten vectors immediately after reading ---
    if name == "sbert":
        # Adjust 'sbert_vector' to whatever the column name is in your specific data
        col_to_flatten = "sbert_vector" if "sbert_vector" in df.columns else "features"
        df = flatten_vectors(df, col_to_flatten)
    elif name == "tfidf":
        # If TF-IDF is stored as a dense vector list, flatten it too
        if "tfidf_vector" in df.columns:
            df = flatten_vectors(df, "tfidf_vector")
            
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

    # 1. Load and Merge
    print("Loading and cleaning datasets...")
    dfs = {
        "length": read_and_clean_df(args.length, "length"),
        "sentiment": read_and_clean_df(args.sentiment, "sentiment"),
        "tfidf": read_and_clean_df(args.tfidf, "tfidf"),
        "sbert": read_and_clean_df(args.sbert, "sbert"),
        "capital": read_and_clean_df(args.capital_letters, "capital_letters")
    }

    merged = dfs["length"]
    for name in ["sentiment", "tfidf", "sbert", "capital"]:
        print(f"Merging {name} features...")
        merged = merged.merge(dfs[name], on=KEYS, how="inner", suffixes=(None, '_drop'))
        
        # Drop duplicate key columns if they exist
        cols_to_drop = [c for c in merged.columns if c.endswith('_drop')]
        if cols_to_drop:
            merged.drop(columns=cols_to_drop, inplace=True)

    # 2. Splitting Logic (60/15/15/10)
    print("Splitting data into 60/15/15/10...")
    
    if 'review_year' in merged.columns:
        merged = merged.sort_values('review_year')
        
    df_rest, df_deploy = train_test_split(merged, test_size=0.10, shuffle=False)
    df_train, df_val_test = train_test_split(df_rest, test_size=0.333, random_state=42)
    df_val, df_test = train_test_split(df_val_test, test_size=0.50, random_state=42)

    # 3. Save to subfolders
    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "deployment": df_deploy
    }

    for name, df in splits.items():
        split_path = os.path.join(args.out, name)
        os.makedirs(split_path, exist_ok=True)
        # index=False is important for Parquet compatibility
        df.to_parquet(os.path.join(split_path, "data.parquet"), index=False)
        print(f"Saved {name} split: {df.shape}")

    print(f"Success! Final merged shape: {merged.shape}")

if __name__ == "__main__":
    main()