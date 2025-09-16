from __future__ import annotations
import os, contextlib

@contextlib.contextmanager
def maybe_mlflow_run(enabled: bool, experiment: str | None = None, run_name: str | None = None):
    try:
        if not enabled:
            yield None
            return
        import mlflow
        if experiment:
            mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name) as run:
            yield run
    except Exception as e:
        # Gracefully degrade if MLflow isn't installed/configured
        print(f"[mlflow] disabled or unavailable: {e}")
        yield None

def log_params(params: dict, client=None):
    try:
        import mlflow
        mlflow.log_params(params)
    except Exception:
        pass

def log_metrics(metrics: dict, step: int | None = None, client=None):
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass

def log_artifact(path: str, client=None):
    try:
        import mlflow
        mlflow.log_artifact(path)
    except Exception:
        pass
