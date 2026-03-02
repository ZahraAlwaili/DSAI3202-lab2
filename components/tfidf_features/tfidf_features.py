import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--model_out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1,2))
    X = vectorizer.fit_transform(df['reviewText'])
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out()).to_parquet(os.path.join(args.out, "data.parquet"))
    os.makedirs(args.model_out, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(args.model_out, "tfidf_vectorizer.pkl"))
    print("TF-IDF features created.")

if __name__ == "__main__":
    main()