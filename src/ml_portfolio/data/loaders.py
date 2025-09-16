from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_sample(data_dir: str | Path, filename: str = "processed/sample.csv") -> pd.DataFrame:
    return load_csv(Path(data_dir) / filename)
