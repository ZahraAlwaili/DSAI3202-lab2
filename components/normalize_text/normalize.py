import argparse
import os
import pandas as pd
import re

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()                       # lowercase
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"\d+", "", text)           # remove numbers
    text = re.sub(r"[^\w\s]", "", text)       # remove punctuation
    text = text.strip()                        # trim whitespace
    return text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_parquet(args.data)

    # normalize review text
    df['reviewText'] = df['reviewText'].apply(normalize_text)

    # filter short/empty reviews
    df = df[df['reviewText'].str.len() >= 10]

    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)
    print("Normalized rows:", len(df))

if __name__ == "__main__":
    main()