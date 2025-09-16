# ML Portfolio

A curated repository of ML learning: notebooks, datasets, and clean, testable Python modules.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
Or, with conda:
```bash
conda env create -f environment.yml
conda activate ml-portfolio
```

## Typical workflow
1. Put data in `data/raw/` (or use DVC/git-lfs).
2. Explore in `notebooks/01_exploratory/`.
3. Build features & models in `src/`.
4. Run pipelines via `scripts/` and track results in `experiments/`.
5. Add tests in `tests/` and keep CI green.

## Structure
- `notebooks/` learning & experiments
- `src/` reusable code
- `scripts/` CLI entrypoints
- `data/` versioned externally (DVC or git-lfs)
- `experiments/` run metadata & reports
- `tests/` unit tests

## Tooling
- Formatting: black, isort
- Linting: ruff
- Tests: pytest
- Optional: DVC for data, MLflow for experiments

---

### Projects index
Add links to your best notebooks and reports here.


## Deep Learning & MLflow (Optional)

Install extras:
```bash
pip install -r requirements-deep.txt
```

### PyTorch
```bash
python scripts/train_pytorch.py --config src/ml_portfolio/configs/deep_pytorch.yaml
```

### TensorFlow (Keras)
```bash
python scripts/train_tensorflow.py --config src/ml_portfolio/configs/deep_tensorflow.yaml
```

### MLflow Tracking
Set your tracking URI (local or remote) and runs will log params/metrics when enabled in config:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
# Enable in config under: mlflow.enabled: true
```

See `README-ADDONS.md` for details.


### Local MLflow with Docker

A minimal MLflow tracking server is provided in `docker/compose-mlflow.yml`.

Run locally:
```bash
cd docker
docker compose -f compose-mlflow.yml up
```

This starts MLflow UI at http://localhost:5000 with SQLite backend and local `mlruns/` artifact store.


### Run a local MLflow server with Docker (optional)

From the `docker/` folder:
```bash
cd docker
docker compose -f compose-mlflow.yml up -d
# visit http://localhost:5000
```

Then, in a new terminal, point your runs to this server:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

Artifacts and runs are persisted under `docker/mlflow/`.
