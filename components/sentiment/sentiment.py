import argparse
import os
import pandas as pd
import nltk
import glob

# Essential for cloud environment
nltk.download('vader_lexicon') 
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Robust way to find the parquet file in the input folder
    input_files = glob.glob(os.path.join(args.data, "*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {args.data}")
    
    df = pd.read_parquet(input_files[0])
    
    sia = SentimentIntensityAnalyzer()
    
    def sentiment_scores(text):
        # Handle potential nulls in text
        if pd.isna(text):
            return pd.Series([0.0, 0.0, 1.0, 0.0])
        scores = sia.polarity_scores(str(text))
        return pd.Series([scores['pos'], scores['neg'], scores['neu'], scores['compound']])
    
    # Apply sentiment analysis
    df[['sentiment_pos','sentiment_neg','sentiment_neu','sentiment_compound']] = df['reviewText'].apply(sentiment_scores)
    
    # Save output
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print(f"Success: Sentiment features created for {len(df)} rows.")

if __name__ == "__main__":
    main()