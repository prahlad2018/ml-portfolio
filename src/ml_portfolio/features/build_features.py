import pandas as pd

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if set(['feat1','feat2']).issubset(df.columns):
        df = df.copy()
        df['feat1_x_feat2'] = df['feat1'] * df['feat2']
    return df
