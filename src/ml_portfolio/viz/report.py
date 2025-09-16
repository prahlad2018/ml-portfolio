from pathlib import Path
from ..utils.io import save_json

def save_eval_report(metrics: dict, out_dir: str | Path, name: str = "report.json"):
    out_path = Path(out_dir) / name
    save_json(metrics, out_path)
    return str(out_path)
