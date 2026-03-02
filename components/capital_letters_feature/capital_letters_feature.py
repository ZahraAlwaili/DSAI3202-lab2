import argparse
import pandas as pd
import os

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
    parser.add_argument("--data", type=str, required=True, help="Input data folder containing CSV")
    parser.add_argument("--out", type=str, required=True, help="Output folder for feature CSV")
    parser.add_argument("--text_column", type=str, default="review_text", help="Name of text column")
    args = parser.parse_args()

    # Load input data (assume single CSV in folder)
    input_files = [f for f in os.listdir(args.data) if f.endswith(".csv")]
    if not input_files:
        raise FileNotFoundError("No CSV file found in input data folder.")
    df = pd.read_csv(os.path.join(args.data, input_files[0]))

    # Compute caps_ratio
    df["caps_ratio"] = df[args.text_column].fillna("").apply(caps_ratio)

    # Create output folder
    os.makedirs(args.out, exist_ok=True)
    output_path = os.path.join(args.out, "data.csv")
    df.to_csv(output_path, index=False)
    print(f"Caps ratio feature added successfully! Output saved to {output_path}")

if __name__ == "__main__":
    main()
    