import argparse
import pandas as pd
import os
import glob

def caps_ratio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    total_letters = sum(c.isalpha() for c in text)
    if total_letters == 0:
        return 0.0
    uppercase_letters = sum(c.isupper() for c in text)
    return uppercase_letters / total_letters

def main():
    parser = argparse.ArgumentParser(description="Capital Letters Ratio Feature")
    parser.add_argument("--data", type=str, required=True, help="Input data folder")
    parser.add_argument("--out", type=str, required=True, help="Output folder")
    parser.add_argument("--text_column", type=str, default="reviewText", help="Name of text column")
    args = parser.parse_args()

    # Look for Parquet files instead of CSV
    input_files = glob.glob(os.path.join(args.data, "*.parquet"))
    
    if not input_files:
        raise FileNotFoundError(f"No Parquet files found in {args.data}")
    
    # Read Parquet
    df = pd.read_parquet(input_files[0])

    # Compute caps_ratio (using reviewText to match your sentiment script)
    # If your normalization script changes the column name, ensure it matches here
    target_col = args.text_column if args.text_column in df.columns else "reviewText"
    df["caps_ratio"] = df[target_col].fillna("").apply(caps_ratio)

    # Create output folder
    os.makedirs(args.out, exist_ok=True)
    # Save as Parquet to keep things consistent across the pipeline
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path)
    
    print(f"Caps ratio feature added successfully! Output saved to {output_path}")

if __name__ == "__main__":
    main()