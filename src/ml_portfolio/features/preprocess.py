from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def numeric_scaler(num_cols):
    return ("num", StandardScaler(), num_cols)

def build_preprocess(num_cols=None):
    transformers = []
    if num_cols:
        transformers.append(numeric_scaler(num_cols))
    ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
    return Pipeline(steps=[("preprocess", ct)])
