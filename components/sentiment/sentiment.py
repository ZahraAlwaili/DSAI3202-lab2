import argparse
import os
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_parquet(args.data)
    sia = SentimentIntensityAnalyzer()
    
    def sentiment_scores(text):
        scores = sia.polarity_scores(text)
        return pd.Series([scores['pos'], scores['neg'], scores['neu'], scores['compound']])
    
    df[['sentiment_pos','sentiment_neg','sentiment_neu','sentiment_compound']] = df['reviewText'].apply(sentiment_scores)
    os.makedirs(args.out, exist_ok=True)
    df.to_parquet(os.path.join(args.out, "data.parquet"))
    print("Sentiment features created for", len(df), "rows.")

if __name__ == "__main__":
    main()