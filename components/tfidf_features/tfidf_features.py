import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Input folder with parquet")
    parser.add_argument("--out", type=str, required=True, help="Output folder for features")
    parser.add_argument("--model_out", type=str, required=True, help="Output folder for vectorizer")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Read the input parquet file from the folder
    input_files = glob.glob(os.path.join(args.data, "*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {args.data}")
    df = pd.read_parquet(input_files[0])

    # 2. Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    # Ensure reviewText exists, fallback to review_text if needed
    text_col = 'reviewText' if 'reviewText' in df.columns else 'review_text'
    X = vectorizer.fit_transform(df[text_col].fillna(""))

    # 3. Create Feature DataFrame
    tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # 4. CRITICAL FIX: Attach the Keys (asin, reviewerID) back to the features
    # We reset_index to ensure the concat alignment is perfect
    keys_df = df[['asin', 'reviewerID']].reset_index(drop=True)
    final_df = pd.concat([keys_df, tfidf_df], axis=1)

    # 5. Save Data
    os.makedirs(args.out, exist_ok=True)
    final_df.to_parquet(os.path.join(args.out, "data.parquet"), index=False)

    # 6. Save Model
    os.makedirs(args.model_out, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(args.model_out, "tfidf_vectorizer.pkl"))
    
    print(f"TF-IDF features created. Shape: {final_df.shape}")

if __name__ == "__main__":
    main()