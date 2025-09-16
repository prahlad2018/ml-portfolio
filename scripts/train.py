import argparse, yaml
from pathlib import Path
import pandas as pd
from src.ml_portfolio.utils.seed import set_seed
from src.ml_portfolio.utils.io import save_joblib, ensure_dir
from src.ml_portfolio.models.registry import get_model
from src.ml_portfolio.models.train import crossval_score_estimator
from src.ml_portfolio.models.evaluate import evaluate_binary
from sklearn.model_selection import train_test_split

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(Path(cfg["paths"]["data_dir"]) / "processed" / "sample.csv")
    y = df[cfg["dataset"]["target"]]
    X = df.drop(columns=[cfg["dataset"]["target"]])

    stratify = y if cfg["dataset"].get("stratify", True) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["dataset"]["test_size"], stratify=stratify, random_state=cfg["seed"]
    )

    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    mean, std = crossval_score_estimator(
        model, X_tr, y_tr, cv_folds=cfg["train"]["cv_folds"], scoring=cfg["train"]["scoring"], seed=cfg["seed"]
    )
    print(f"CV {cfg['train']['scoring']}: {mean:.4f} ± {std:.4f}")

    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(X_te)[:, 1]
    metrics = evaluate_binary(y_te, y_proba)
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")

    ensure_dir(cfg['paths']['models_dir'])
    save_joblib(model, Path(cfg['paths']['models_dir']) / f"{cfg['model']['name']}.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/ml_portfolio/configs/config.yaml")
    args = ap.parse_args()
    main(args.config)
