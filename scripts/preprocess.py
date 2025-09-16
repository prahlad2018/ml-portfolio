import pandas as pd
from pathlib import Path
from src.ml_portfolio.features.build_features import add_interactions
from src.ml_portfolio.utils.io import ensure_dir

def main():
    inp = Path("data/processed/sample.csv")
    df = pd.read_csv(inp)
    df = add_interactions(df)
    ensure_dir("data/processed")
    df.to_csv("data/processed/sample_features.csv", index=False)
    print("Saved -> data/processed/sample_features.csv")

if __name__ == "__main__":
    main()
