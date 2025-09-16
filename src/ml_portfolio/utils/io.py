from pathlib import Path
import joblib, json

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_joblib(obj, path: str | Path):
    path = Path(path); ensure_dir(path.parent)
    joblib.dump(obj, path)
    return str(path)

def save_json(obj, path: str | Path):
    path = Path(path); ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return str(path)
