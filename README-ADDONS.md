# Add-ons: PyTorch, TensorFlow, MLflow

These files extend the base scaffold with:
- `scripts/train_pytorch.py` — simple MLP on tabular data
- `scripts/train_tensorflow.py` — simple Keras MLP
- `src/ml_portfolio/trackers/mlflow_utils.py` — optional MLflow tracking

Install extras (optional):
```bash
pip install -r requirements-deep.txt
# or: pip install mlflow torch tensorflow  # customize versions for your OS/GPU
```

Run examples:
```bash
python scripts/train_pytorch.py --config src/ml_portfolio/configs/deep_pytorch.yaml
python scripts/train_tensorflow.py --config src/ml_portfolio/configs/deep_tensorflow.yaml
```
