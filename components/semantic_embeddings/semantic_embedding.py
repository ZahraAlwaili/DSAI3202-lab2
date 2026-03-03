import argparse
import pandas as pd
import os
import glob
import torch
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="review_text")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    
    # Locate parquet file
    input_files = glob.glob(os.path.join(args.input_data, "*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {args.input_data}")
    
    # Read data
    df = pd.read_parquet(input_files[0])
    
    # Identify column
    target_col = args.text_column if args.text_column in df.columns else "reviewText"
    df[target_col] = df[target_col].fillna("")
    
    # Load Model
    print(f"Loading model: {args.model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model_name, device=device)
    
    # Extract Embeddings (Batch size 32)
    print(f"Processing {len(df)} rows...")
    embeddings = model.encode(
        df[target_col].tolist(), 
        batch_size=32, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    embedding_df = pd.DataFrame(
        embeddings,
        columns=[f"bert_embedding_{i}" for i in range(embeddings.shape[1])]
    )
    
    # Merge and Save
    result_df = pd.concat([df.reset_index(drop=True), embedding_df.reset_index(drop=True)], axis=1)
    os.makedirs(args.output_data, exist_ok=True)
    result_df.to_parquet(os.path.join(args.output_data, "data.parquet"))
    print("Success! Embeddings merged and saved.")

if __name__ == "__main__":
    main()