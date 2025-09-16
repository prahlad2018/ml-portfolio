import pandas as pd
from src.ml_portfolio.features.build_features import add_interactions

def test_add_interactions():
    df = pd.DataFrame({'feat1':[1,2], 'feat2':[3,4]})
    out = add_interactions(df)
    assert 'feat1_x_feat2' in out.columns
