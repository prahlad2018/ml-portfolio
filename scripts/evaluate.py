import argparse, yaml, joblib, pandas as pd
from pathlib import Path
from src.ml_portfolio.viz.report import save_eval_report

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_path = Path(cfg["paths"]["models_dir"]) / f"{cfg['model']['name']}.joblib"
    model = joblib.load(model_path)
    df = pd.read_csv(Path(cfg["paths"]["data_dir"]) / "processed" / "sample.csv")
    y = df[cfg["dataset"]["target"]]
    X = df.drop(columns=[cfg["dataset"]["target"]])

    y_proba = model.predict_proba(X)[:, 1]
    from src.ml_portfolio.models.evaluate import evaluate_binary
    metrics = evaluate_binary(y, y_proba)

    out = save_eval_report(metrics, cfg["paths"]["experiments_dir"], "last_eval.json")
    print("Saved report ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/ml_portfolio/configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
