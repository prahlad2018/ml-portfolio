import argparse, yaml, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.ml_portfolio.trackers.mlflow_utils import maybe_mlflow_run, log_params, log_metrics
from src.ml_portfolio.utils.seed import set_seed

def build_mlp(in_dim, hidden_dims, lr=1e-3, dropout=0.0):
    import tensorflow as tf
    from tensorflow import keras
    inputs = keras.Input(shape=(in_dim,))
    x = inputs
    for h in hidden_dims:
        x = keras.layers.Dense(h, activation="relu")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc")])
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

    val_split = cfg["train"].get("val_split", 0.2)
    model = build_mlp(X.shape[1], cfg["train"]["hidden_dims"], lr=cfg["train"]["lr"], dropout=cfg["train"].get("dropout", 0.0))

    run_kwargs = cfg.get("mlflow", {})
    with maybe_mlflow_run(run_kwargs.get("enabled", False), run_kwargs.get("experiment"), run_kwargs.get("run_name")):
        log_params({
            "model": "tf_mlp",
            "hidden_dims": str(cfg["train"]["hidden_dims"]),
            "epochs": cfg["train"]["epochs"],
            "batch_size": cfg["train"]["batch_size"],
            "lr": cfg["train"]["lr"],
            "dropout": cfg["train"].get("dropout", 0.0),
        })
        history = model.fit(
            X_tr, y_tr,
            validation_split=val_split,
            epochs=cfg["train"]["epochs"],
            batch_size=cfg["train"]["batch_size"],
            verbose=2
        )
        # Evaluate
        y_proba = model.predict(X_te, verbose=0).ravel()
        test_auc = roc_auc_score(y_te, y_proba)
        print(f"Test ROC-AUC: {test_auc:.4f}")
        log_metrics({"test_auc": float(test_auc)}, step=cfg["train"]["epochs"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/ml_portfolio/configs/deep_tensorflow.yaml")
    args = ap.parse_args()
    main(args.config)
