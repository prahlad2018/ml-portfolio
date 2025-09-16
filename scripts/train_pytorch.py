import argparse, yaml, numpy as np, pandas as pd
from pathlib import Path
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.ml_portfolio.trackers.mlflow_utils import maybe_mlflow_run, log_params, log_metrics
from src.ml_portfolio.utils.seed import set_seed

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_loop(model, X_tr, y_tr, X_val, y_val, epochs=20, batch_size=32, lr=1e-3, weight_decay=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32); y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32); y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n = len(X_tr_t)
    for epoch in range(1, epochs+1):
        model.train()
        perm = torch.randperm(n)
        X_tr_t, y_tr_t = X_tr_t[perm], y_tr_t[perm]
        for i in range(0, n, batch_size):
            xb = X_tr_t[i:i+batch_size].to(device); yb = y_tr_t[i:i+batch_size].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()

        # Eval
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device)).cpu().numpy()
            val_proba = 1/(1+np.exp(-val_logits))
            val_auc = roc_auc_score(y_val, val_proba)
        print(f"Epoch {epoch:02d}  val_auc={val_auc:.4f}")
        log_metrics({"val_auc": float(val_auc)}, step=epoch)
    return model

def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    df = pd.read_csv(Path(cfg["paths"]["data_dir"]) / "processed" / "sample.csv")
    y = df[cfg["dataset"]["target"]].values.astype(np.float32)
    X = df.drop(columns=[cfg["dataset"]["target"]]).values.astype(np.float32)

    stratify = y if cfg["dataset"].get("stratify", True) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["dataset"]["test_size"], stratify=stratify, random_state=cfg["seed"]
    )

    # val split from train
    val_split = cfg["train"].get("val_split", 0.2)
    n_val = int(len(X_tr) * val_split)
    X_val, y_val = X_tr[:n_val], y_tr[:n_val]
    X_sub, y_sub = X_tr[n_val:], y_tr[n_val:]

    model = MLP(in_dim=X.shape[1], hidden_dims=cfg["train"]["hidden_dims"])

    run_kwargs = cfg.get("mlflow", {})
    with maybe_mlflow_run(run_kwargs.get("enabled", False), run_kwargs.get("experiment"), run_kwargs.get("run_name")):
        log_params({
            "model": "pytorch_mlp",
            "hidden_dims": str(cfg["train"]["hidden_dims"]),
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch_size"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"].get("weight_decay", 0.0),
        })
        trained = train_loop(
            model, X_sub, y_sub, X_val, y_val,
            epochs=cfg["train"]["epochs"],
            batch_size=cfg["train"]["batch_size"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )
        # Final test AUC
        trained.eval()
        with torch.no_grad():
            logits = trained(torch.tensor(X_te, dtype=torch.float32)).numpy()
            proba = 1/(1+np.exp(-logits))
            test_auc = roc_auc_score(y_te, proba)
        print(f"Test ROC-AUC: {test_auc:.4f}")
        log_metrics({"test_auc": float(test_auc)}, step=cfg["train"]["epochs"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/ml_portfolio/configs/deep_pytorch.yaml")
    args = ap.parse_args()
    main(args.config)
